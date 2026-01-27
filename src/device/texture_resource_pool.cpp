#include "texture_resource_pool.h"
#include "../utils/debug.h"

// 初始化资源池
void TextureResourcePool::Initialize(size_t poolSize) {
    resources.resize(poolSize);
    DBG_LOGF("xxxxxxxxxxxxxx Initialize resources.size=%d poolSize=%d", resources.size(), poolSize);
}

// 获取空闲资源
ID3D11Texture2D* TextureResourcePool::GetAvailableResource() {
    for (auto& resource : resources) {
        if (!resource.used) {
            resource.used = true;
            return resource.pTexture;
        }
    }
    return nullptr; // 没有可用资源
}

// 返回资源
void TextureResourcePool::ReturnResource(ID3D11Texture2D* texture) {
    for (auto& resource : resources) {
        if (resource.pTexture == texture) {
            resource.used = false;
            break;
        }
    }
}

// 设置资源纹理
void TextureResourcePool::SetTexture(size_t index, ID3D11Texture2D* texture) {
    // DBG_LOG("SetTexture -1: index=%d, texture=%p resources.size()=%d\n", index, texture, resources.size());
    if (index < resources.size()) {
        resources[index].pTexture = texture;
        resources[index].used = false;
       // DBG_LOG("SetTexture -2: index=%d, texture=%p\n", index, texture);
    }
}