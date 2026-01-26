#pragma once
#include <vector>
#include <d3d11.h>

// 资源结构体
struct TextureResource {
    ID3D11Texture2D* pTexture;
    bool used;
    TextureResource() : pTexture(nullptr), used(false) {}
};

// 资源池类
class TextureResourcePool {
private:
    std::vector<TextureResource> resources;

public:
    // 初始化资源池
    void Initialize(size_t poolSize);

    // 获取空闲资源
    ID3D11Texture2D* GetAvailableResource();

    // 返回资源
    void ReturnResource(ID3D11Texture2D* texture);

    // 设置资源纹理
    void SetTexture(size_t index, ID3D11Texture2D* texture);
};
