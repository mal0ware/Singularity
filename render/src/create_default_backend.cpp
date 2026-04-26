// render/src/create_default_backend.cpp
//
// Platform-appropriate backend factory. Each SINGULARITY_BACKEND_<API> option
// gates a forward declaration + a call into the concrete factory. The first
// declared backend that returns a non-null pointer wins, in native-first
// order: Metal on Apple, Vulkan on Windows, CUDA last-resort (offline only).
//
// This file is deliberately one-of-a-kind for the render library so the
// library can stay INTERFACE when nothing is compiled in and STATIC when
// something is. See render/CMakeLists.txt.

#include <memory>

#include "render_backend.hpp"

namespace singularity {

#if defined(SINGULARITY_HAS_METAL)
namespace metal {
std::unique_ptr<RenderBackend> create_metal_backend();
}
#endif

#if defined(SINGULARITY_HAS_VULKAN)
namespace vulkan {
std::unique_ptr<RenderBackend> create_vulkan_backend();
}
#endif

#if defined(SINGULARITY_HAS_WEBGPU)
namespace webgpu {
std::unique_ptr<RenderBackend> create_webgpu_backend();
}
#endif

std::unique_ptr<RenderBackend> create_default_backend() {
#if defined(SINGULARITY_HAS_METAL)
    if (auto b = metal::create_metal_backend())
        return b;
#endif
#if defined(SINGULARITY_HAS_VULKAN)
    if (auto b = vulkan::create_vulkan_backend())
        return b;
#endif
#if defined(SINGULARITY_HAS_WEBGPU)
    if (auto b = webgpu::create_webgpu_backend())
        return b;
#endif
    return nullptr;
}

}  // namespace singularity
