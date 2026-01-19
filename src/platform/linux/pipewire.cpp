/**
 * @file src/platform/linux/pipewire.cpp
 * @brief PipeWire portal capture for Wayland (KDE, GNOME, etc.)
 */

#include <gio/gio.h>
#include <gio/gunixfdlist.h>
#include <pipewire/pipewire.h>
#include <spa/param/video/format-utils.h>

#include <spa/pod/builder.h>
#include <spa/param/buffers.h>
#include <drm_fourcc.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <fstream>

#include "src/logging.h"
#include "src/platform/common.h"
#include "src/video.h"
#include "src/config.h"
#include "graphics.h"
#include "vaapi.h"

using namespace std::literals;

namespace platf::pipewire {

  constexpr auto PORTAL_DEST = "org.freedesktop.portal.Desktop";
  constexpr auto PORTAL_PATH = "/org/freedesktop/portal/desktop";
  constexpr auto PORTAL_SCREENCAST = "org.freedesktop.portal.ScreenCast";
  constexpr auto PORTAL_REQUEST = "org.freedesktop.portal.Request";

  static std::string g_restore_token;
  static std::mutex g_token_mutex;

  static std::string get_token_path() {
    return platf::appdata().string() + "/pipewire_token";
  }

  static void load_restore_token() {
    if (!config::video.pipewire.persist_session) return;
    std::ifstream f(get_token_path());
    if (f) {
      std::getline(f, g_restore_token);
    }
  }

  static void save_restore_token() {
    if (!config::video.pipewire.persist_session) return;
    std::ofstream f(get_token_path());
    if (f) {
      f << g_restore_token;
    }
  }

  struct frame_t {
    std::vector<uint8_t> data;
    int dmabuf_fd = -1;
    uint32_t stride = 0;
    uint32_t offset = 0;
    uint64_t modifier = DRM_FORMAT_MOD_INVALID;
    bool is_dmabuf = false;
    bool valid = false;

  };

  struct pw_state_t {
    pw_thread_loop *loop = nullptr;
    pw_context *context = nullptr;
    pw_core *core = nullptr;
    pw_stream *stream = nullptr;
    spa_hook stream_listener {};
    uint32_t node_id = 0;
    int pw_fd = -1;
    std::mutex mtx;
    std::condition_variable cv;
    frame_t current_frame;
    std::atomic<bool> running {true};
    std::atomic<int> ref_count {0};
    uint32_t width = 0;
    uint32_t height = 0;
    bool initialized = false;

    void cleanup() {
      running = false;
      cv.notify_all();
      if (loop) pw_thread_loop_stop(loop);
      if (stream) { pw_stream_destroy(stream); stream = nullptr; }
      if (core) { pw_core_disconnect(core); core = nullptr; }
      if (context) { pw_context_destroy(context); context = nullptr; }
      if (loop) { pw_thread_loop_destroy(loop); loop = nullptr; }
      if (pw_fd >= 0) { close(pw_fd); pw_fd = -1; }
      if (current_frame.dmabuf_fd >= 0) { close(current_frame.dmabuf_fd); current_frame.dmabuf_fd = -1; }
      current_frame.valid = false;
      current_frame.data.clear();
      initialized = false;
      width = height = 0;
    }
  };

  static pw_state_t g_pw_state;
  static std::mutex g_pw_mutex;

  struct portal_session_t {
    GDBusConnection *conn = nullptr;
    char *session_handle = nullptr;
    uint32_t node_id = 0;
    int pw_fd = -1;
    bool success = false;
    GMainLoop *loop = nullptr;

    ~portal_session_t() {
      g_free(session_handle);
      if (pw_fd >= 0) close(pw_fd);
      if (conn) g_object_unref(conn);
      if (loop) g_main_loop_unref(loop);
    }
  };

  static bool g_dmabuf_negotiated = false;
  static uint64_t g_dmabuf_modifier = DRM_FORMAT_MOD_INVALID;

  static void on_stream_param_changed(void *data, uint32_t id, const struct spa_pod *param) {
    auto *state = static_cast<pw_state_t*>(data);
    if (!param || id != SPA_PARAM_Format) return;

    struct spa_video_info_raw vinfo;
    if (spa_format_video_raw_parse(param, &vinfo) < 0) return;

    state->width = vinfo.size.width;
    state->height = vinfo.size.height;

    g_dmabuf_negotiated = spa_pod_find_prop(param, nullptr, SPA_FORMAT_VIDEO_modifier) != nullptr;
    if (g_dmabuf_negotiated) {
      g_dmabuf_modifier = vinfo.modifier;
    }

    BOOST_LOG(info) << "PipeWire: "sv << state->width << "x"sv << state->height
                    << " "sv << vinfo.framerate.num << "/"sv << vinfo.framerate.denom << "fps"sv
                    << (g_dmabuf_negotiated ? " DMA-BUF"sv : " SHM"sv);

    uint8_t buf[1024];
    spa_pod_builder b = SPA_POD_BUILDER_INIT(buf, sizeof(buf));
    uint32_t buffer_types = g_dmabuf_negotiated ? (1 << SPA_DATA_DmaBuf) : ((1 << SPA_DATA_MemFd) | (1 << SPA_DATA_MemPtr));

    const spa_pod *params[2];
    int n_params = 0;

    params[n_params++] = static_cast<spa_pod*>(spa_pod_builder_add_object(&b,
      SPA_TYPE_OBJECT_ParamBuffers, SPA_PARAM_Buffers,
      SPA_PARAM_BUFFERS_dataType, SPA_POD_CHOICE_FLAGS_Int(buffer_types)));

    pw_stream_update_params(state->stream, params, n_params);
  }

  static void on_stream_process(void *data) {
    auto *state = static_cast<pw_state_t*>(data);

    struct pw_buffer *buf = pw_stream_dequeue_buffer(state->stream);
    if (!buf) return;

    struct spa_buffer *spa_buf = buf->buffer;
    struct spa_data *d = &spa_buf->datas[0];
    std::lock_guard<std::mutex> lock(state->mtx);

    if (state->current_frame.dmabuf_fd >= 0) {
      close(state->current_frame.dmabuf_fd);
      state->current_frame.dmabuf_fd = -1;
    }

    bool got_data = false;
    uint32_t stride = d->chunk->stride > 0 ? d->chunk->stride : state->width * 4;

    if (d->type == SPA_DATA_DmaBuf && d->fd >= 0) {
      state->current_frame.dmabuf_fd = fcntl(d->fd, F_DUPFD_CLOEXEC, 0);
      state->current_frame.stride = stride;
      state->current_frame.offset = d->chunk->offset;
      state->current_frame.modifier = g_dmabuf_modifier;
      state->current_frame.is_dmabuf = true;
      state->current_frame.data.clear();
      got_data = true;
    } else if ((d->type == SPA_DATA_MemPtr || d->type == SPA_DATA_MemFd) && d->data && d->chunk->size > 0) {
      state->current_frame.data.resize(d->chunk->size);
      memcpy(state->current_frame.data.data(), d->data, d->chunk->size);
      state->current_frame.stride = stride;
      state->current_frame.is_dmabuf = false;
      got_data = true;
    }

    if (got_data) {
      state->current_frame.valid = true;
      state->cv.notify_one();
    }

    pw_stream_queue_buffer(state->stream, buf);
  }

  static void on_stream_state_changed(void *data, enum pw_stream_state old, enum pw_stream_state st, const char *err_msg) {
    auto *state = static_cast<pw_state_t*>(data);
    if (st == PW_STREAM_STATE_STREAMING) {
      state->running = true;
    } else if (st == PW_STREAM_STATE_UNCONNECTED || st == PW_STREAM_STATE_ERROR) {
      state->running = false;
      state->initialized = false;
    }
  }

  static const pw_stream_events stream_events = {
    .version = PW_VERSION_STREAM_EVENTS,
    .state_changed = on_stream_state_changed,
    .param_changed = on_stream_param_changed,
    .process = on_stream_process,
  };

  static void on_response(GDBusConnection *conn, const gchar *sender, const gchar *path,
                          const gchar *interface, const gchar *signal, GVariant *params, gpointer data) {
    auto *session = static_cast<portal_session_t*>(data);

    uint32_t response;
    GVariant *results;
    g_variant_get(params, "(u@a{sv})", &response, &results);

    if (response != 0) {
      BOOST_LOG(error) << "Portal response error: "sv << response;
      g_main_loop_quit(session->loop);
      g_variant_unref(results);
      return;
    }

    const char *handle;
    if (g_variant_lookup(results, "session_handle", "&s", &handle)) {
      session->session_handle = g_strdup(handle);
    }

    const char *restore_token;
    if (g_variant_lookup(results, "restore_token", "&s", &restore_token)) {
      std::lock_guard<std::mutex> lock(g_token_mutex);
      g_restore_token = restore_token;
      save_restore_token();
      BOOST_LOG(info) << "Saved restore token for future sessions"sv;
    }

    GVariant *streams;
    if (g_variant_lookup(results, "streams", "@a(ua{sv})", &streams)) {
      GVariantIter iter;
      g_variant_iter_init(&iter, streams);
      uint32_t node_id;
      GVariant *props;
      if (g_variant_iter_next(&iter, "(u@a{sv})", &node_id, &props)) {
        session->node_id = node_id;
        BOOST_LOG(info) << "PipeWire node_id: "sv << node_id;
        g_variant_unref(props);
      }
      g_variant_unref(streams);
    }

    g_variant_unref(results);
    session->success = true;
    g_main_loop_quit(session->loop);
  }

  static char* get_request_path(GDBusConnection *conn, const char *token) {
    const char *sender = g_dbus_connection_get_unique_name(conn);
    char *sender_escaped = g_strdup(sender + 1);
    for (char *p = sender_escaped; *p; p++) {
      if (*p == '.') *p = '_';
    }
    char *path = g_strdup_printf("/org/freedesktop/portal/desktop/request/%s/%s", sender_escaped, token);
    g_free(sender_escaped);
    return path;
  }

  static bool portal_call(portal_session_t &session, const char *method, const char *token, GVariant *params) {
    GError *gerr = nullptr;
    char *request_path = get_request_path(session.conn, token);

    session.success = false;
    guint signal_id = g_dbus_connection_signal_subscribe(
      session.conn, PORTAL_DEST, PORTAL_REQUEST, "Response",
      request_path, nullptr, G_DBUS_SIGNAL_FLAGS_NO_MATCH_RULE,
      on_response, &session, nullptr);

    g_dbus_connection_call_sync(
      session.conn, PORTAL_DEST, PORTAL_PATH, PORTAL_SCREENCAST,
      method, params, nullptr, G_DBUS_CALL_FLAGS_NONE, -1, nullptr, &gerr);

    if (gerr) {
      BOOST_LOG(error) << method << " failed: "sv << gerr->message;
      g_error_free(gerr);
      g_dbus_connection_signal_unsubscribe(session.conn, signal_id);
      g_free(request_path);
      return false;
    }

    g_main_loop_run(session.loop);
    g_dbus_connection_signal_unsubscribe(session.conn, signal_id);
    g_free(request_path);
    return session.success;
  }

  static bool portal_create_session(portal_session_t &session) {
    GError *gerr = nullptr;
    session.conn = g_bus_get_sync(G_BUS_TYPE_SESSION, nullptr, &gerr);
    if (!session.conn) {
      BOOST_LOG(error) << "Failed to connect to D-Bus: "sv << gerr->message;
      g_error_free(gerr);
      return false;
    }

    session.loop = g_main_loop_new(nullptr, FALSE);
    const char *token = "sunshine_session";

    GVariantBuilder opts;
    g_variant_builder_init(&opts, G_VARIANT_TYPE("a{sv}"));
    g_variant_builder_add(&opts, "{sv}", "handle_token", g_variant_new_string(token));
    g_variant_builder_add(&opts, "{sv}", "session_handle_token", g_variant_new_string("sunshine"));

    if (!portal_call(session, "CreateSession", token, g_variant_new("(a{sv})", &opts))) return false;
    return session.session_handle != nullptr;
  }

  static bool portal_select_sources(portal_session_t &session) {
    const char *token = "sunshine_select";

    GVariantBuilder opts;
    g_variant_builder_init(&opts, G_VARIANT_TYPE("a{sv}"));
    g_variant_builder_add(&opts, "{sv}", "handle_token", g_variant_new_string(token));
    g_variant_builder_add(&opts, "{sv}", "types", g_variant_new_uint32(1));
    g_variant_builder_add(&opts, "{sv}", "cursor_mode", g_variant_new_uint32(2));
    g_variant_builder_add(&opts, "{sv}", "persist_mode", g_variant_new_uint32(2));

    {
      std::lock_guard<std::mutex> lock(g_token_mutex);
      if (!g_restore_token.empty()) {
        g_variant_builder_add(&opts, "{sv}", "restore_token", g_variant_new_string(g_restore_token.c_str()));
      }
    }

    return portal_call(session, "SelectSources", token, g_variant_new("(oa{sv})", session.session_handle, &opts));
  }

  static bool portal_start(portal_session_t &session) {
    const char *token = "sunshine_start";

    GVariantBuilder opts;
    g_variant_builder_init(&opts, G_VARIANT_TYPE("a{sv}"));
    g_variant_builder_add(&opts, "{sv}", "handle_token", g_variant_new_string(token));

    if (!portal_call(session, "Start", token, g_variant_new("(osa{sv})", session.session_handle, "", &opts))) return false;
    return session.node_id != 0;
  }

  static bool portal_open_pipewire(portal_session_t &session) {
    GError *gerr = nullptr;
    GUnixFDList *fd_list = nullptr;

    GVariantBuilder opts;
    g_variant_builder_init(&opts, G_VARIANT_TYPE("a{sv}"));

    GVariant *result = g_dbus_connection_call_with_unix_fd_list_sync(
      session.conn, PORTAL_DEST, PORTAL_PATH, PORTAL_SCREENCAST,
      "OpenPipeWireRemote", g_variant_new("(oa{sv})", session.session_handle, &opts),
      G_VARIANT_TYPE("(h)"), G_DBUS_CALL_FLAGS_NONE, -1,
      nullptr, &fd_list, nullptr, &gerr);

    if (gerr) {
      BOOST_LOG(error) << "OpenPipeWireRemote failed: "sv << gerr->message;
      g_error_free(gerr);
      return false;
    }

    int32_t fd_index;
    g_variant_get(result, "(h)", &fd_index);
    session.pw_fd = g_unix_fd_list_get(fd_list, fd_index, nullptr);

    g_variant_unref(result);
    g_object_unref(fd_list);
    return session.pw_fd >= 0;
  }

  static int init_global_pw_state() {
    std::lock_guard<std::mutex> lock(g_pw_mutex);

    if (g_pw_state.initialized && g_pw_state.stream) {
      enum pw_stream_state state = pw_stream_get_state(g_pw_state.stream, nullptr);

      if (state == PW_STREAM_STATE_PAUSED) {
        pw_thread_loop_lock(g_pw_state.loop);
        pw_stream_set_active(g_pw_state.stream, true);
        pw_thread_loop_unlock(g_pw_state.loop);

        for (int i = 0; i < 20 && state != PW_STREAM_STATE_STREAMING; i++) {
          std::this_thread::sleep_for(50ms);
          state = pw_stream_get_state(g_pw_state.stream, nullptr);
        }
      }

      if (state == PW_STREAM_STATE_STREAMING) {
        g_pw_state.ref_count++;
        g_pw_state.running = true;
        g_pw_state.current_frame.valid = false;
        return 0;
      }

      g_pw_state.cleanup();
    }

    pw_init(nullptr, nullptr);
    load_restore_token();

    portal_session_t portal;
    if (!portal_create_session(portal)) return -1;
    if (!portal_select_sources(portal)) return -1;
    if (!portal_start(portal)) return -1;
    if (!portal_open_pipewire(portal)) return -1;

    g_pw_state.node_id = portal.node_id;
    g_pw_state.pw_fd = portal.pw_fd;
    portal.pw_fd = -1;

    g_pw_state.loop = pw_thread_loop_new("sunshine-pw", nullptr);
    if (!g_pw_state.loop) {
      BOOST_LOG(error) << "Failed to create PipeWire thread loop"sv;
      return -1;
    }

    g_pw_state.context = pw_context_new(pw_thread_loop_get_loop(g_pw_state.loop), nullptr, 0);
    if (!g_pw_state.context) {
      BOOST_LOG(error) << "Failed to create PipeWire context"sv;
      g_pw_state.cleanup();
      return -1;
    }

    g_pw_state.core = pw_context_connect_fd(g_pw_state.context, fcntl(g_pw_state.pw_fd, F_DUPFD_CLOEXEC, 0), nullptr, 0);
    if (!g_pw_state.core) {
      BOOST_LOG(error) << "Failed to connect PipeWire core"sv;
      g_pw_state.cleanup();
      return -1;
    }

    auto props = pw_properties_new(
      PW_KEY_MEDIA_TYPE, "Video",
      PW_KEY_MEDIA_CATEGORY, "Capture",
      PW_KEY_MEDIA_ROLE, "Screen",
      PW_KEY_NODE_LATENCY, "1/60",
      nullptr);

    g_pw_state.stream = pw_stream_new(g_pw_state.core, "sunshine-capture", props);
    if (!g_pw_state.stream) {
      BOOST_LOG(error) << "Failed to create PipeWire stream"sv;
      g_pw_state.cleanup();
      return -1;
    }

    pw_stream_add_listener(g_pw_state.stream, &g_pw_state.stream_listener, &stream_events, &g_pw_state);

    uint8_t buffer[4096];
    spa_pod_builder b = SPA_POD_BUILDER_INIT(buffer, sizeof(buffer));

    static struct spa_rectangle min_size = SPA_RECTANGLE(1, 1);
    static struct spa_rectangle max_size = SPA_RECTANGLE(4096, 4096);
    static struct spa_rectangle def_size = SPA_RECTANGLE(1920, 1080);
    static struct spa_fraction min_rate = SPA_FRACTION(0, 1);
    static struct spa_fraction max_rate = SPA_FRACTION(144, 1);
    static struct spa_fraction def_rate = SPA_FRACTION(60, 1);

    const spa_pod *params[2];
    int n_params = 0;

    bool use_dmabuf = config::video.encoder.empty() || config::video.encoder == "vaapi";

    if (use_dmabuf) {
      spa_pod_frame dmabuf_frame, modifier_frame;
      spa_pod_builder_push_object(&b, &dmabuf_frame, SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat);
      spa_pod_builder_add(&b,
        SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
        SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
        SPA_FORMAT_VIDEO_format, SPA_POD_CHOICE_ENUM_Id(3, SPA_VIDEO_FORMAT_BGRx, SPA_VIDEO_FORMAT_BGRA, SPA_VIDEO_FORMAT_RGBx),
        0);
      spa_pod_builder_prop(&b, SPA_FORMAT_VIDEO_modifier, SPA_POD_PROP_FLAG_MANDATORY | SPA_POD_PROP_FLAG_DONT_FIXATE);
      spa_pod_builder_push_choice(&b, &modifier_frame, SPA_CHOICE_Enum, 0);
      spa_pod_builder_long(&b, DRM_FORMAT_MOD_INVALID);
      spa_pod_builder_long(&b, DRM_FORMAT_MOD_INVALID);
      spa_pod_builder_pop(&b, &modifier_frame);
      spa_pod_builder_add(&b,
        SPA_FORMAT_VIDEO_size, SPA_POD_CHOICE_RANGE_Rectangle(&def_size, &min_size, &max_size),
        SPA_FORMAT_VIDEO_framerate, SPA_POD_CHOICE_RANGE_Fraction(&def_rate, &min_rate, &max_rate),
        0);
      params[n_params++] = static_cast<spa_pod*>(spa_pod_builder_pop(&b, &dmabuf_frame));
    }

    params[n_params++] = static_cast<spa_pod*>(spa_pod_builder_add_object(&b,
      SPA_TYPE_OBJECT_Format, SPA_PARAM_EnumFormat,
      SPA_FORMAT_mediaType, SPA_POD_Id(SPA_MEDIA_TYPE_video),
      SPA_FORMAT_mediaSubtype, SPA_POD_Id(SPA_MEDIA_SUBTYPE_raw),
      SPA_FORMAT_VIDEO_format, SPA_POD_CHOICE_ENUM_Id(3, SPA_VIDEO_FORMAT_BGRx, SPA_VIDEO_FORMAT_BGRA, SPA_VIDEO_FORMAT_RGBx),
      SPA_FORMAT_VIDEO_size, SPA_POD_CHOICE_RANGE_Rectangle(&def_size, &min_size, &max_size),
      SPA_FORMAT_VIDEO_framerate, SPA_POD_CHOICE_RANGE_Fraction(&def_rate, &min_rate, &max_rate)));

    pw_thread_loop_lock(g_pw_state.loop);
    int ret = pw_stream_connect(g_pw_state.stream, PW_DIRECTION_INPUT, g_pw_state.node_id,
        static_cast<pw_stream_flags>(PW_STREAM_FLAG_AUTOCONNECT | PW_STREAM_FLAG_MAP_BUFFERS),
        params, n_params);
    pw_thread_loop_unlock(g_pw_state.loop);

    if (ret < 0) {
      BOOST_LOG(error) << "Failed to connect PipeWire stream"sv;
      g_pw_state.cleanup();
      return -1;
    }

    pw_thread_loop_start(g_pw_state.loop);
    g_pw_state.running = true;
    g_pw_state.ref_count = 1;
    g_pw_state.initialized = true;

    for (int i = 0; i < 50 && g_pw_state.width == 0; i++) {
      std::this_thread::sleep_for(100ms);
    }

    if (g_pw_state.width == 0) {
      BOOST_LOG(error) << "Timeout waiting for PipeWire stream"sv;
      g_pw_state.cleanup();
      return -1;
    }

    BOOST_LOG(info) << "PipeWire capture: "sv << g_pw_state.width << "x"sv << g_pw_state.height;
    return 0;
  }

  class display_pw_t: public display_t {
  public:
    display_pw_t(mem_type_e mem_type): mem_type(mem_type) {}
    ~display_pw_t() { g_pw_state.ref_count--; }

    int init(const std::string &display_name, const ::video::config_t &config) {
      if (init_global_pw_state() < 0) return -1;
      width = g_pw_state.width;
      height = g_pw_state.height;
      frame_duration = std::chrono::nanoseconds(1000000000 / config.framerate);
      return 0;
    }

    capture_e capture(const push_captured_image_cb_t &push_captured_image_cb,
                      const pull_free_image_cb_t &pull_free_image_cb, bool *cursor) override {
      auto next_frame_time = std::chrono::steady_clock::now();
      while (g_pw_state.running) {
        std::shared_ptr<platf::img_t> img_out;
        if (!pull_free_image_cb(img_out)) return capture_e::interrupted;

        bool frame_copied = false;
        {
          std::unique_lock<std::mutex> lock(g_pw_state.mtx);
          if (!g_pw_state.cv.wait_for(lock, 100ms, [] { return g_pw_state.current_frame.valid || !g_pw_state.running; })) {
            continue;
          }
          if (!g_pw_state.running) return capture_e::interrupted;
          if (!g_pw_state.current_frame.valid) continue;
          if (!g_pw_state.current_frame.is_dmabuf && g_pw_state.current_frame.data.empty()) {
            g_pw_state.current_frame.valid = false;
            continue;
          }

          if (img_out->data && !g_pw_state.current_frame.data.empty()) {
            auto *img = img_out.get();
            size_t copy_size = std::min(g_pw_state.current_frame.data.size(), (size_t)(img->row_pitch * img->height));
            memcpy(img->data, g_pw_state.current_frame.data.data(), copy_size);
            g_pw_state.current_frame.valid = false;
            frame_copied = true;
          }
        }

        if (frame_copied) {
          auto now = std::chrono::steady_clock::now();
          if (now < next_frame_time) {
            std::this_thread::sleep_until(next_frame_time);
          }
          next_frame_time = std::chrono::steady_clock::now() + frame_duration;
          
          if (!push_captured_image_cb(std::move(img_out), true)) {
            return capture_e::ok;
          }
        }
      }
      return capture_e::ok;
    }

    std::shared_ptr<img_t> alloc_img() override {
      auto img = std::make_shared<img_t>();
      img->width = width;
      img->height = height;
      img->pixel_pitch = 4;
      img->row_pitch = width * 4;
      img->data = new uint8_t[img->row_pitch * height];
      return img;
    }

    int dummy_img(img_t *img) override { return 0; }

    std::unique_ptr<avcodec_encode_device_t> make_avcodec_encode_device(pix_fmt_e pix_fmt) override {
#ifdef SUNSHINE_BUILD_VAAPI
      if (mem_type == mem_type_e::vaapi) {
        return va::make_avcodec_encode_device(width, height, false);
      }
#endif
      return std::make_unique<avcodec_encode_device_t>();
    }

    bool is_hdr() override { return false; }
    bool get_hdr_metadata(SS_HDR_METADATA &metadata) override { return false; }
    bool variable_framerate() override { return true; }

  private:
    mem_type_e mem_type;
    int width = 0;
    int height = 0;
    std::chrono::nanoseconds frame_duration;
  };

  class display_pw_vram_t: public display_t {
  public:
    ~display_pw_vram_t() { g_pw_state.ref_count--; }

    int init(const std::string &display_name, const ::video::config_t &config) {
      if (init_global_pw_state() < 0) return -1;

      width = g_pw_state.width;
      height = g_pw_state.height;
      frame_duration = std::chrono::nanoseconds(1000000000 / config.framerate);

      auto render_device = config::video.adapter_name.empty() ? "/dev/dri/renderD128" : config::video.adapter_name.c_str();
      int fd = open(render_device, O_RDWR);
      if (fd < 0) {
        BOOST_LOG(error) << "Failed to open render device: "sv << render_device;
        return -1;
      }

      egl_display = egl::make_display(gbm::create_device(fd));
      if (!egl_display) {
        close(fd);
        return -1;
      }

      auto ctx_opt = egl::make_ctx(egl_display.get());
      if (!ctx_opt) return -1;
      ctx = std::move(*ctx_opt);

      return 0;
    }

    capture_e capture(const push_captured_image_cb_t &push_captured_image_cb,
                      const pull_free_image_cb_t &pull_free_image_cb, bool *cursor) override {
      auto next_frame_time = std::chrono::steady_clock::now();
      while (g_pw_state.running) {
        std::shared_ptr<platf::img_t> img_out;
        if (!pull_free_image_cb(img_out)) return capture_e::interrupted;

        bool got_frame = false;
        {
          std::unique_lock<std::mutex> lock(g_pw_state.mtx);
          if (g_pw_state.cv.wait_for(lock, 100ms, [] { return g_pw_state.current_frame.valid || !g_pw_state.running; })) {
            if (!g_pw_state.running) return capture_e::interrupted;
            if (g_pw_state.current_frame.valid) {
              got_frame = fill_image(img_out);
            }
          }
        }

        if (got_frame) {
          auto now = std::chrono::steady_clock::now();
          if (now < next_frame_time) {
            std::this_thread::sleep_until(next_frame_time);
          }
          next_frame_time = std::chrono::steady_clock::now() + frame_duration;
        }

        if (!push_captured_image_cb(std::move(img_out), got_frame)) return capture_e::ok;
      }
      return capture_e::ok;
    }

    std::shared_ptr<img_t> alloc_img() override {
      auto img = std::make_shared<egl::img_descriptor_t>();
      img->width = width;
      img->height = height;
      img->sequence = 0;
      img->data = nullptr;
      std::fill_n(img->sd.fds, 4, -1);
      return img;
    }

    int dummy_img(img_t *img) override { return 0; }

    std::unique_ptr<avcodec_encode_device_t> make_avcodec_encode_device(pix_fmt_e pix_fmt) override {
#ifdef SUNSHINE_BUILD_VAAPI
      return va::make_avcodec_encode_device(width, height, 0, 0, true);
#endif
      return std::make_unique<avcodec_encode_device_t>();
    }

    bool is_hdr() override { return false; }
    bool get_hdr_metadata(SS_HDR_METADATA &metadata) override { return false; }
    bool variable_framerate() override { return true; }

  private:
    bool fill_image(std::shared_ptr<platf::img_t> &img_out) {
      if (!g_pw_state.current_frame.is_dmabuf || g_pw_state.current_frame.dmabuf_fd < 0) {
        g_pw_state.current_frame.valid = false;
        return false;
      }

      auto *img = static_cast<egl::img_descriptor_t*>(img_out.get());
      img->reset();
      img->sd.width = width;
      img->sd.height = height;
      img->sd.fds[0] = g_pw_state.current_frame.dmabuf_fd;
      g_pw_state.current_frame.dmabuf_fd = -1;
      img->sd.pitches[0] = g_pw_state.current_frame.stride;
      img->sd.offsets[0] = g_pw_state.current_frame.offset;
      img->sd.fourcc = DRM_FORMAT_XRGB8888;
      img->sd.modifier = g_pw_state.current_frame.modifier;
      std::fill_n(img->sd.fds + 1, 3, -1);
      img->sequence = ++sequence;
      img->frame_timestamp = std::chrono::steady_clock::now();
      g_pw_state.current_frame.valid = false;
      return true;
    }

    int width = 0;
    int height = 0;
    uint64_t sequence = 0;
    std::chrono::nanoseconds frame_duration;
    egl::display_t egl_display;
    egl::ctx_t ctx;
  };

  std::shared_ptr<display_t> display(mem_type_e hwdevice_type, const std::string &display_name, const ::video::config_t &config) {
    if (hwdevice_type == mem_type_e::vaapi) {
      auto disp = std::make_shared<display_pw_vram_t>();
      if (disp->init(display_name, config) == 0 && g_dmabuf_negotiated) {
        BOOST_LOG(info) << "PipeWire DMA-BUF zero-copy capture"sv;
        return disp;
      }
      g_pw_state.ref_count--;
    }

    auto disp = std::make_shared<display_pw_t>(hwdevice_type);
    if (disp->init(display_name, config) < 0) return nullptr;
    return disp;
  }

  std::vector<std::string> display_names() {
    return {"pipewire"};
  }

  bool verify() {
    GError *gerr = nullptr;
    auto conn = g_bus_get_sync(G_BUS_TYPE_SESSION, nullptr, &gerr);
    if (!conn) {
      if (gerr) g_error_free(gerr);
      return false;
    }

    auto result = g_dbus_connection_call_sync(
      conn, PORTAL_DEST, PORTAL_PATH, "org.freedesktop.DBus.Properties",
      "Get", g_variant_new("(ss)", PORTAL_SCREENCAST, "AvailableSourceTypes"),
      nullptr, G_DBUS_CALL_FLAGS_NONE, 1000, nullptr, &gerr);

    g_object_unref(conn);
    if (gerr) {
      g_error_free(gerr);
      return false;
    }
    g_variant_unref(result);
    return true;
  }

}
