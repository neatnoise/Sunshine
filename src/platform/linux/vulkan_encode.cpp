/**
 * @file src/platform/linux/vulkan_encode.cpp
 * @brief FFmpeg Vulkan encoder - zero-copy via Vulkan→DMA-BUF→EGL.
 */

#include <fcntl.h>
#include <drm_fourcc.h>
#include <gbm.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_vulkan.h>
#include <libavutil/hwcontext_drm.h>
#include <libavutil/imgutils.h>
}

#include <vulkan/vulkan.h>

#include "vulkan_encode.h"
#include "graphics.h"
#include "src/logging.h"
#include "misc.h"

using namespace std::literals;

namespace vk {

  class vk_vram_t: public platf::avcodec_encode_device_t {
  public:
    ~vk_vram_t() {
      if (sw_frame) av_frame_free(&sw_frame);
      if (drm_frame) av_frame_free(&drm_frame);
      if (gbm_bo_y) gbm_bo_destroy(gbm_bo_y);
      if (gbm_bo_uv) gbm_bo_destroy(gbm_bo_uv);
    }

    int init(int in_width, int in_height, file_t &&render_device) {
      file = std::move(render_device);
      width = in_width;
      height = in_height;

      if (!gbm::create_device) {
        BOOST_LOG(warning) << "libgbm not initialized"sv;
        return -1;
      }

      this->data = (void *) vulkan_init_avcodec_hardware_input_buffer;

      gbm.reset(gbm::create_device(file.el));
      if (!gbm) {
        BOOST_LOG(error) << "Couldn't create GBM device"sv;
        return -1;
      }

      display = egl::make_display(gbm.get());
      if (!display) return -1;

      auto ctx_opt = egl::make_ctx(display.get());
      if (!ctx_opt) return -1;
      ctx = std::move(*ctx_opt);

      return 0;
    }

    int set_frame(AVFrame *frame, AVBufferRef *hw_frames_ctx_buf) override {
      this->hwframe.reset(frame);
      this->frame = frame;
      this->hw_frames_ctx = hw_frames_ctx_buf;

      auto *frames_ctx = (AVHWFramesContext *) hw_frames_ctx_buf->data;
      auto *dev_ctx = (AVHWDeviceContext *) frames_ctx->device_ref->data;
      auto *vk_dev_ctx = (AVVulkanDeviceContext *) dev_ctx->hwctx;

      vk_dev = vk_dev_ctx->act_dev;

      // Load Vulkan extension functions
      if (!vkGetMemoryFdKHR_fn) {
        vkGetMemoryFdKHR_fn = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(vk_dev, "vkGetMemoryFdKHR");
        vkGetImageSubresourceLayout_fn = (PFN_vkGetImageSubresourceLayout)vkGetDeviceProcAddr(vk_dev, "vkGetImageSubresourceLayout");
      }

      // Create sws for RGB->NV12 conversion
      auto sws_opt = egl::sws_t::make(frame->width, frame->height, frame->width, frame->height, frames_ctx->sw_format);
      if (!sws_opt) return -1;
      sws = std::move(*sws_opt);

      return 0;
    }

    void apply_colorspace() override {}
    
    void init_hwframes(AVHWFramesContext *frames) override {
      frames->initial_pool_size = 4;
    }

    int convert(platf::img_t &img) override {
      auto &descriptor = (egl::img_descriptor_t &) img;

      if (descriptor.sequence == 0) {
        rgb = egl::create_blank(img);
      } else if (descriptor.sequence > sequence) {
        sequence = descriptor.sequence;
        rgb = egl::rgb_t {};
        auto rgb_opt = egl::import_source(display.get(), descriptor.sd);
        if (!rgb_opt) return -1;
        rgb = std::move(*rgb_opt);
      }

      // Get Vulkan frame
      if (!frame->buf[0]) {
        if (av_hwframe_get_buffer(hw_frames_ctx, frame, 0) < 0) {
          BOOST_LOG(error) << "Failed to get Vulkan frame"sv;
          return -1;
        }
        nv12_imported = false;  // New frame, need to re-import
      }

      // Setup Vulkan→EGL interop if needed
      if (!nv12_imported) {
        if (!setup_vulkan_egl_interop()) {
          // Fallback to CPU path
          return convert_cpu_fallback(descriptor);
        }
        nv12_imported = true;
      }

      // Render RGB→NV12 directly into Vulkan memory via EGL
      sws.load_vram(descriptor, 0, 0, rgb->tex[0]);
      sws.convert(nv12->buf);
      
      // Ensure EGL is done before Vulkan uses the frame
      gl::ctx.Finish();

      return 0;
    }

  private:
    bool setup_vulkan_egl_interop() {
      if (!vkGetMemoryFdKHR_fn || !vkGetImageSubresourceLayout_fn) {
        return false;
      }

      AVVkFrame *vk_frame = (AVVkFrame *) frame->data[0];
      if (!vk_frame) {
        return false;
      }

      std::array<file_t, 4> fds;
      std::array<egl::surface_descriptor_t, 4> sds;

      // Export Y and UV planes
      for (int i = 0; i < 2; i++) {
        // Export memory to DMA-BUF
        VkMemoryGetFdInfoKHR fd_info = {};
        fd_info.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
        fd_info.memory = vk_frame->mem[i];
        fd_info.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

        int fd = -1;
        VkResult res = vkGetMemoryFdKHR_fn(vk_dev, &fd_info, &fd);
        if (res != VK_SUCCESS || fd < 0) {
          return false;
        }
        fds[i].el = fd;

        // Get image layout
        VkImageSubresource subres = {};
        subres.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        subres.mipLevel = 0;
        subres.arrayLayer = 0;

        VkSubresourceLayout layout;
        vkGetImageSubresourceLayout_fn(vk_dev, vk_frame->img[i], &subres, &layout);

        // Setup surface descriptor
        sds[i].fourcc = (i == 0) ? DRM_FORMAT_R8 : DRM_FORMAT_GR88;
        sds[i].width = frame->width >> (i ? 1 : 0);
        sds[i].height = frame->height >> (i ? 1 : 0);
        sds[i].pitches[0] = layout.rowPitch;
        sds[i].offsets[0] = layout.offset;
        sds[i].modifier = DRM_FORMAT_MOD_LINEAR;
        sds[i].fds[0] = fds[i].el;
        
        BOOST_LOG(info) << "Plane " << i << ": " << sds[i].width << "x" << sds[i].height << " pitch=" << layout.rowPitch;
      }

      // Import into EGL
      auto nv12_opt = egl::import_target(display.get(), std::move(fds), sds[0], sds[1]);
      if (!nv12_opt) {
        BOOST_LOG(warning) << "Failed to import Vulkan frame into EGL"sv;
        return false;
      }
      nv12 = std::move(*nv12_opt);

      BOOST_LOG(info) << "Vulkan-EGL zero-copy interop established"sv;
      return true;
    }

    int convert_cpu_fallback(egl::img_descriptor_t &descriptor) {
      // Try GBM path first (GPU render, then map to CPU)
      if (try_drm_transfer(descriptor)) {
        return 0;
      }

      // Ultimate fallback: EGL texture readback
      if (!nv12_cpu_init) {
        auto nv12_opt = egl::create_target(frame->width, frame->height, AV_PIX_FMT_NV12);
        if (!nv12_opt) return -1;
        nv12_cpu = std::move(*nv12_opt);
        nv12_cpu_init = true;
        
        sw_frame = av_frame_alloc();
        sw_frame->format = AV_PIX_FMT_NV12;
        sw_frame->width = frame->width;
        sw_frame->height = frame->height;
        av_frame_get_buffer(sw_frame, 32);
        y_buf.resize(frame->width * frame->height);
        uv_buf.resize(frame->width * frame->height / 2);
      }

      sws.load_vram(descriptor, 0, 0, rgb->tex[0]);
      sws.convert(nv12_cpu->buf);

      return cpu_readback();
    }
    
    bool try_drm_transfer(egl::img_descriptor_t &descriptor) {
      // Create GBM buffer-backed NV12 if not done
      if (!gbm_nv12_init) {
        auto *gbm_dev = (gbm_device *)gbm.get();
        
        // Create GBM buffer for Y plane
        gbm_bo_y = gbm_bo_create(gbm_dev, frame->width, frame->height, GBM_FORMAT_R8, GBM_BO_USE_RENDERING | GBM_BO_USE_LINEAR);
        if (!gbm_bo_y) {
          BOOST_LOG(warning) << "Failed to create GBM Y buffer"sv;
          return false;
        }
        
        // Create GBM buffer for UV plane  
        gbm_bo_uv = gbm_bo_create(gbm_dev, frame->width / 2, frame->height / 2, GBM_FORMAT_GR88, GBM_BO_USE_RENDERING | GBM_BO_USE_LINEAR);
        if (!gbm_bo_uv) {
          BOOST_LOG(warning) << "Failed to create GBM UV buffer"sv;
          gbm_bo_destroy(gbm_bo_y);
          gbm_bo_y = nullptr;
          return false;
        }
        
        // Get DMA-BUF fds
        int fd_y = gbm_bo_get_fd(gbm_bo_y);
        int fd_uv = gbm_bo_get_fd(gbm_bo_uv);
        if (fd_y < 0 || fd_uv < 0) {
          BOOST_LOG(warning) << "Failed to get GBM DMA-BUF fds"sv;
          return false;
        }
        
        // Import into EGL as render target
        std::array<file_t, 4> fds;
        fds[0].el = dup(fd_y);
        fds[1].el = dup(fd_uv);
        
        egl::surface_descriptor_t sd_y = {}, sd_uv = {};
        sd_y.fourcc = DRM_FORMAT_R8;
        sd_y.width = frame->width;
        sd_y.height = frame->height;
        sd_y.pitches[0] = gbm_bo_get_stride(gbm_bo_y);
        sd_y.offsets[0] = 0;
        sd_y.modifier = DRM_FORMAT_MOD_LINEAR;
        sd_y.fds[0] = fds[0].el;
        sd_y.fds[1] = sd_y.fds[2] = sd_y.fds[3] = -1;
        
        sd_uv.fourcc = DRM_FORMAT_GR88;
        sd_uv.width = frame->width / 2;
        sd_uv.height = frame->height / 2;
        sd_uv.pitches[0] = gbm_bo_get_stride(gbm_bo_uv);
        sd_uv.offsets[0] = 0;
        sd_uv.modifier = DRM_FORMAT_MOD_LINEAR;
        sd_uv.fds[0] = fds[1].el;
        sd_uv.fds[1] = sd_uv.fds[2] = sd_uv.fds[3] = -1;
        
        auto nv12_opt = egl::import_target(display.get(), std::move(fds), sd_y, sd_uv);
        if (!nv12_opt) {
          BOOST_LOG(warning) << "Failed to import GBM buffers to EGL"sv;
          return false;
        }
        nv12_gbm = std::move(*nv12_opt);
        
        // Allocate sw_frame for transfer
        sw_frame = av_frame_alloc();
        sw_frame->format = AV_PIX_FMT_NV12;
        sw_frame->width = frame->width;
        sw_frame->height = frame->height;
        av_frame_get_buffer(sw_frame, 32);
        
        BOOST_LOG(info) << "GBM-backed NV12 render target created: " << frame->width << "x" << frame->height;
        gbm_nv12_init = true;
      }
      
      // Render RGB→NV12 to GBM-backed textures
      sws.load_vram(descriptor, 0, 0, rgb->tex[0]);
      sws.convert(nv12_gbm->buf);
      gl::ctx.Finish();
      
      // Map GBM buffers and copy to sw_frame
      uint32_t y_stride;
      void *y_map = gbm_bo_map(gbm_bo_y, 0, 0, frame->width, frame->height, GBM_BO_TRANSFER_READ, &y_stride, &gbm_map_data_y);
      if (!y_map) {
        return false;
      }
      
      for (int y = 0; y < frame->height; y++) {
        memcpy(sw_frame->data[0] + y * sw_frame->linesize[0], (uint8_t*)y_map + y * y_stride, frame->width);
      }
      gbm_bo_unmap(gbm_bo_y, gbm_map_data_y);
      
      uint32_t uv_stride;
      void *uv_map = gbm_bo_map(gbm_bo_uv, 0, 0, frame->width / 2, frame->height / 2, GBM_BO_TRANSFER_READ, &uv_stride, &gbm_map_data_uv);
      if (!uv_map) {
        return false;
      }
      
      for (int y = 0; y < frame->height / 2; y++) {
        memcpy(sw_frame->data[1] + y * sw_frame->linesize[1], (uint8_t*)uv_map + y * uv_stride, frame->width);
      }
      gbm_bo_unmap(gbm_bo_uv, gbm_map_data_uv);
      
      return av_hwframe_transfer_data(frame, sw_frame, 0) >= 0;
    }
    
    int cpu_readback() {
      gl::ctx.BindTexture(GL_TEXTURE_2D, nv12_cpu->tex[0]);
      gl::ctx.PixelStorei(GL_PACK_ALIGNMENT, 1);
      gl::ctx.GetTexImage(GL_TEXTURE_2D, 0, GL_RED, GL_UNSIGNED_BYTE, y_buf.data());
      for (int y = 0; y < frame->height; y++)
        memcpy(sw_frame->data[0] + y * sw_frame->linesize[0], y_buf.data() + y * frame->width, frame->width);

      gl::ctx.BindTexture(GL_TEXTURE_2D, nv12_cpu->tex[1]);
      gl::ctx.GetTexImage(GL_TEXTURE_2D, 0, GL_RG, GL_UNSIGNED_BYTE, uv_buf.data());
      for (int y = 0; y < frame->height / 2; y++)
        memcpy(sw_frame->data[1] + y * sw_frame->linesize[1], uv_buf.data() + y * frame->width, frame->width);

      gl::ctx.BindTexture(GL_TEXTURE_2D, 0);
      return av_hwframe_transfer_data(frame, sw_frame, 0);
    }

    static int vulkan_init_avcodec_hardware_input_buffer(platf::avcodec_encode_device_t *, AVBufferRef **hw_device_buf) {
      BOOST_LOG(info) << "Creating Vulkan hardware device"sv;
      return av_hwdevice_ctx_create(hw_device_buf, AV_HWDEVICE_TYPE_VULKAN, nullptr, nullptr, 0);
    }

    int width = 0, height = 0;
    AVBufferRef *hw_frames_ctx = nullptr;
    AVFrame *sw_frame = nullptr;
    AVFrame *drm_frame = nullptr;
    frame_t hwframe;

    file_t file;
    gbm::gbm_t gbm;
    egl::display_t display;
    egl::ctx_t ctx;
    egl::sws_t sws;
    egl::nv12_t nv12;       // Zero-copy target
    egl::nv12_t nv12_cpu;   // CPU fallback target
    egl::nv12_t nv12_gbm;   // GBM-backed target
    egl::rgb_t rgb;
    egl::img_descriptor_t descriptor_cache;
    std::uint64_t sequence = 0;
    std::vector<uint8_t> y_buf;
    std::vector<uint8_t> uv_buf;
    bool nv12_imported = false;
    bool nv12_cpu_init = false;
    bool gbm_nv12_init = false;
    
    gbm_bo *gbm_bo_y = nullptr;
    gbm_bo *gbm_bo_uv = nullptr;
    void *gbm_map_data_y = nullptr;
    void *gbm_map_data_uv = nullptr;

    VkDevice vk_dev = VK_NULL_HANDLE;
    PFN_vkGetMemoryFdKHR vkGetMemoryFdKHR_fn = nullptr;
    PFN_vkGetImageSubresourceLayout vkGetImageSubresourceLayout_fn = nullptr;
  };

  int vulkan_init_avcodec_hardware_input_buffer(platf::avcodec_encode_device_t *, AVBufferRef **hw_device_buf) {
    BOOST_LOG(info) << "Creating Vulkan hardware device"sv;
    return av_hwdevice_ctx_create(hw_device_buf, AV_HWDEVICE_TYPE_VULKAN, nullptr, nullptr, 0);
  }

  bool validate() {
    if (!avcodec_find_encoder_by_name("h264_vulkan") && !avcodec_find_encoder_by_name("hevc_vulkan"))
      return false;
    AVBufferRef *dev = nullptr;
    if (av_hwdevice_ctx_create(&dev, AV_HWDEVICE_TYPE_VULKAN, nullptr, nullptr, 0) < 0)
      return false;
    av_buffer_unref(&dev);
    BOOST_LOG(info) << "Vulkan Video encoding available"sv;
    return true;
  }

  std::unique_ptr<platf::avcodec_encode_device_t> make_avcodec_encode_device_vram(int w, int h, int, int) {
    file_t file = open("/dev/dri/renderD128", O_RDWR);
    if (file.el < 0) {
      BOOST_LOG(error) << "Failed to open render device"sv;
      return nullptr;
    }
    auto dev = std::make_unique<vk_vram_t>();
    if (dev->init(w, h, std::move(file)) < 0) return nullptr;
    return dev;
  }

  std::unique_ptr<platf::avcodec_encode_device_t> make_avcodec_encode_device_ram(int, int) {
    return nullptr;
  }

}  // namespace vk
