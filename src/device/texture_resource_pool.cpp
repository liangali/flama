#include "texture_resource_pool.h"
#include "../utils/debug.h"

// Initialize resource pool
void TextureResourcePool::Initialize(size_t poolSize) {
    resources.resize(poolSize);
    DBG_LOGF("Initialize resources.size=%d poolSize=%d", resources.size(), poolSize);
}

// Get available resource
ID3D11Texture2D* TextureResourcePool::GetAvailableResource() {
    for (auto& resource : resources) {
        if (!resource.used) {
            resource.used = true;
            return resource.pTexture;
        }
    }
    return nullptr; // No available resources
}

// Return resource
void TextureResourcePool::ReturnResource(ID3D11Texture2D* texture) {
    for (auto& resource : resources) {
        if (resource.pTexture == texture) {
            resource.used = false;
            break;
        }
    }
}

// Release all textures and clear the pool
void TextureResourcePool::ReleaseAll() {
    for (auto& resource : resources) {
        if (resource.pTexture) {
            resource.pTexture->Release();
            resource.pTexture = nullptr;
        }
        resource.used = false;
    }
    resources.clear();
}

// Set texture resource
void TextureResourcePool::SetTexture(size_t index, ID3D11Texture2D* texture) {
    if (index < resources.size()) {
        resources[index].pTexture = texture;
        resources[index].used = false;
    }
}
