/**
 * @file src/platform/linux/pipewire.h
 * @brief PipeWire portal capture declarations
 */
#pragma once

#include <memory>
#include <string>
#include <vector>
#include "src/platform/common.h"

namespace platf::pipewire {
  std::shared_ptr<display_t> display(mem_type_e hwdevice_type, const std::string &display_name, const ::video::config_t &config);
  std::vector<std::string> display_names();
  bool verify();
}
