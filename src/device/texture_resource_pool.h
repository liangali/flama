#pragma once
#include <vector>
#include <d3d11.h>

// Texture resource structure
struct TextureResource {
    ID3D11Texture2D* pTexture;
    bool used;
    TextureResource() : pTexture(nullptr), used(false) {}
};

// Texture resource pool class
class TextureResourcePool {
private:
    std::vector<TextureResource> resources;

public:
    // Initialize resource pool
    void Initialize(size_t poolSize);

    // Get available resource
    ID3D11Texture2D* GetAvailableResource();

    // Return resource
    void ReturnResource(ID3D11Texture2D* texture);

    // Set texture resource
    void SetTexture(size_t index, ID3D11Texture2D* texture);
};
