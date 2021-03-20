#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEFAULT_ALIGNED_GENTYPES
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_FORCE_RADIANS
#include <glm/common.hpp>
#include <glm/vec2.hpp>
#include <glm/vec3.hpp>
#include <glm/matrix.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/hash.hpp>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <tiny_obj_loader.h>

#include <iostream>
#include <stdexcept>
#include <cstdlib>
#include <vector>
#include <optional>
#include <set>
#include <fstream>
#include <array>
#include <chrono>
#include <unordered_map>

#include "Camera.h"
#include "VulkanManager.h"

struct Vertex
{
    glm::vec3 pos;
    glm::vec3 color;
    glm::vec2 texCoord;
    glm::vec3 normal;

    bool operator==(const Vertex& other) const
    {
        return pos == other.pos && color == other.color && texCoord == other.texCoord && normal == other.normal;
    }

    static VkVertexInputBindingDescription getBindingDescription()
    {
        VkVertexInputBindingDescription bindingDescription{};
        bindingDescription.binding = 0;
        bindingDescription.stride = sizeof(Vertex);
        bindingDescription.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

        return bindingDescription;
    }

    static std::array<VkVertexInputAttributeDescription, 4> getAttributeDescriptions()
    {
        std::array<VkVertexInputAttributeDescription, 4> attributeDescriptions{};
        attributeDescriptions[0].binding = 0;
        attributeDescriptions[0].location = 0;
        attributeDescriptions[0].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[0].offset = offsetof(Vertex, pos);

        attributeDescriptions[1].binding = 0;
        attributeDescriptions[1].location = 1;
        attributeDescriptions[1].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[1].offset = offsetof(Vertex, color);

        attributeDescriptions[2].binding = 0;
        attributeDescriptions[2].location = 2;
        attributeDescriptions[2].format = VK_FORMAT_R32G32_SFLOAT;
        attributeDescriptions[2].offset = offsetof(Vertex, texCoord);

        attributeDescriptions[3].binding = 0;
        attributeDescriptions[3].location = 3;
        attributeDescriptions[3].format = VK_FORMAT_R32G32B32_SFLOAT;
        attributeDescriptions[3].offset = offsetof(Vertex, normal);

        return attributeDescriptions;
    }
};


namespace std
{
    template<> struct hash<Vertex>
    {
        size_t operator()(Vertex const& vertex) const
        {
            return ((hash<glm::vec3>()(vertex.pos) ^
                (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^
                (hash<glm::vec2>()(vertex.texCoord) << 1);
        }
    };
}

class HelloTriangleApplication {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;

    const std::string MODEL_PATH = "models/viking_room.obj";
    const std::string TEXTURE_PATH = "textures/viking_room.png";

    const std::vector<const char*> validationLayers = 
    {
    "VK_LAYER_KHRONOS_validation"
    };

    const std::vector<const char*> deviceExtensions = 
    {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
    };

#ifdef NDEBUG
    const bool enableValidationLayers = false;
#else
    const bool enableValidationLayers = true;
#endif

    const int MAX_FRAMES_IN_FLIGHT = 3;

    GLFWwindow* window;

    size_t currentFrame = 0;

    struct SwapChainSupportDetails 
    {
        VkSurfaceCapabilitiesKHR capabilities;
        std::vector<VkSurfaceFormatKHR> formats;
        std::vector<VkPresentModeKHR> presentModes;
    };

    VkInstance instance;
    VkPhysicalDevice physicalDevice;
    VkDevice device;
    VkQueue graphicsQueue;
    VkQueue presentQueue;
    VkSurfaceKHR surface;

    VkSwapchainKHR swapChain;
    std::vector<VkImage> swapChainImages;
    std::vector<VkImageView> swapChainImageViews;
    VkFormat swapChainImageFormat;
    VkExtent2D swapChainExtent;

    VkRenderPass imguiRenderPass;
    VkRenderPass renderPass;

    VkDescriptorSetLayout descriptorSetLayout;
    VkDescriptorSetLayout descriptorSetLayoutSecondPass;

    VkPipelineLayout pipelineLayout;
    VkPipelineLayout pipelineLayoutSecondPass;
    VkPipeline graphicsPipeline;
    VkPipeline graphicsPipelineSecondPass;

    std::vector<VkFramebuffer> swapChainFramebuffers;
    std::vector<VkFramebuffer> imguiFramebuffers;

    VkCommandPool commandPool;
    VkDescriptorPool imguiDescriptorPool;
    VkDescriptorPool descriptorPool;
    std::vector<VkDescriptorSet> descriptorSets;
    std::vector<VkDescriptorSet> descriptorSetsSecondPass;

    std::vector<VkCommandBuffer> commandBuffers;

    std::vector<VkSemaphore> imageAvailableSemaphores;
    std::vector<VkSemaphore> renderFinishedSemaphores;
    std::vector<VkFence> inFlightFences;
    std::vector<VkFence> imagesInFlight;

    std::vector<VkBuffer> vertexBuffer;
    std::vector<VkDeviceMemory> vertexBufferMemory;
    std::vector<VkBuffer> vertexBufferLightSource;
    std::vector<VkDeviceMemory> vertexBufferMemoryLightSource;
    std::vector<VkBuffer> vertexBufferSecondPass;
    std::vector<VkDeviceMemory> vertexBufferMemorySecondPass;


    VkBuffer indexBuffer;
    VkDeviceMemory indexBufferMemory;
    VkBuffer indexBufferSecondPass;
    VkDeviceMemory indexBufferMemorySecondPass;

    std::vector<VkBuffer> uniformBuffers;
    std::vector<VkDeviceMemory> uniformBuffersMemory;
    std::vector<VkBuffer> uniformBuffersShaderInfo;
    std::vector<VkDeviceMemory> uniformBuffersShaderInfoMemory;

    VkImage textureImage;
    VkDeviceMemory textureImageMemory;
    VkImageView textureImageView;
    VkSampler textureSampler;

    VkImage depthImage;
    VkDeviceMemory depthImageMemory;
    VkImageView depthImageView;

    VkImage colorImage;
    VkDeviceMemory colorImageMemory;
    VkImageView colorImageView;

    
    VkBuffer stagingBuffer;
    VkDeviceMemory stagingBufferMemory;

    std::vector<Vertex> vertices;
    const std::vector<Vertex> verticesLightSource = {
    {{-1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},},
    {{ 1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},},
    {{ 1.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},},
    {{-1.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},}
    };
    const std::vector<Vertex> verticesSecondPass = {
        {{-1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},},
        {{ 1.0f, 1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},},
        {{ 1.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},},
        {{-1.0f, -1.0f, 0.0f}, {0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f, 0.0f},}
    };
    std::unordered_map<Vertex, uint32_t> uniqueVertices{};

    std::vector<uint32_t> indices;
    std::vector<uint32_t> indicesLightSource = 
    {
    {0,1,2,2,3,0}
    };
    const std::vector<uint32_t> indicesSecondPass = {
        {0,1,2,2,3,0}
    };

    struct UniformBufferObject 
    {
        glm::mat4 view;
        glm::mat4 proj;
        glm::vec3 viewPos;
    };

    struct UniformShaderInfo
    {
        glm::vec3 lightColor;
        glm::vec3 lightPos;
    };

    struct PushConstantInfo
    {
        glm::mat4 model;
    };

    struct Position
    {
        glm::vec3 pos;
        Position(glm::vec3 pos)
            : pos(pos)
        {

        }
    };

    struct LookAt
    {
        glm::vec3 pos;
        LookAt(glm::vec3 pos)
            : pos(pos)
        {

        }
    };

    struct Up
    {
        glm::vec3 pos;
        Up(glm::vec3 pos)
            : pos(pos)
        {

        }
    };

    struct VulkanImage
    {
        VkImage imageHandle;
        VkDeviceMemory imageMemory;
        VkImageView imageView;
    };

    struct BackgroundItem
    {
        glm::vec2 pos;
        int layerLevel;
        int width;
        int height;
        bool initialized;
        VulkanImage vulkanImage;
    };

    void initVulkan() 
    {
        createInstance();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        VulkanHelperFunctions::createCommandPool(commandPool, physicalDevice, device, surface);
        createDescriptorPool();
        createSwapChain();
        initImGui();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        createGraphicsPipeline();
        createDepthResources();
        createFramebuffers();
        VulkanHelperFunctions::createTextureImage(device, physicalDevice, graphicsQueue, commandPool, TEXTURE_PATH, textureImage, textureImageMemory);
        createTextureImageView();
        VulkanHelperFunctions::createTextureSampler(device, physicalDevice, textureSampler);
        loadModel();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
    }

    void initWindow()
    {
        glfwInit();

        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initImGui()
    {
        //create imgui render pass
        VkAttachmentDescription attachment = {};
        attachment.format = swapChainImageFormat;
        attachment.samples = VK_SAMPLE_COUNT_1_BIT;
        attachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        attachment.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        VkAttachmentReference color_attachment = {};
        color_attachment.attachment = 0;
        color_attachment.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        VkSubpassDescription subpass = {};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &color_attachment;
        VkSubpassDependency dependency = {};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;
        VkRenderPassCreateInfo info = {};
        info.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        info.attachmentCount = 1;
        info.pAttachments = &attachment;
        info.subpassCount = 1;
        info.pSubpasses = &subpass;
        info.dependencyCount = 1;
        info.pDependencies = &dependency;
        if (vkCreateRenderPass(device, &info, nullptr, &imguiRenderPass) != VK_SUCCESS) 
        {
            throw std::runtime_error("Could not create Dear ImGui's render pass");
        }

        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();

        // Setup Platform/Renderer bindings
        ImGui_ImplGlfw_InitForVulkan(window, true);
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = instance;
        init_info.PhysicalDevice = physicalDevice;
        init_info.Device = device;
        VulkanHelperFunctions::QueueFamilyIndices indices = VulkanHelperFunctions::findQueueFamilies(physicalDevice, surface);
        init_info.QueueFamily = indices.graphicsFamily.value();
        init_info.Queue = graphicsQueue;
        init_info.PipelineCache = VK_NULL_HANDLE;
        init_info.DescriptorPool = imguiDescriptorPool;
        init_info.Allocator = nullptr;
        init_info.MinImageCount = swapChainImages.size();
        init_info.ImageCount = swapChainImages.size();
        init_info.CheckVkResultFn = nullptr;
        ImGui_ImplVulkan_Init(&init_info, imguiRenderPass);
        VkCommandBuffer command_buffer = VulkanHelperFunctions::beginSingleTimeCommands(device, commandPool);
        ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
        VulkanHelperFunctions::endSingleTimeCommands(device, commandPool, graphicsQueue, command_buffer);

    }

    void createInstance() 
    {
        if (enableValidationLayers && !checkValidationLayerSupport()) 
        {
            throw std::runtime_error("validation layers requested, but not available!");
        }

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;
        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
        if (enableValidationLayers) 
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else 
        {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create instance!");
        }

        /* Print extensions available*/
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
        std::vector<VkExtensionProperties> extensions(extensionCount);
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());
        std::cout << "available extensions:\n";
        for (const auto& extension : extensions) 
        {
            std::cout << '\t' << extension.extensionName << '\n';
        }
    }

    bool checkValidationLayerSupport() 
    {
        uint32_t layerCount;
        vkEnumerateInstanceLayerProperties(&layerCount, nullptr);

        std::vector<VkLayerProperties> availableLayers(layerCount);
        vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

        for (const char* layerName : validationLayers) 
        {
            bool layerFound = false;

            for (const auto& layerProperties : availableLayers) 
            {
                if (strcmp(layerName, layerProperties.layerName) == 0) 
                {
                    layerFound = true;
                    break;
                }
            }

            if (!layerFound) 
            {
                return false;
            }
        }
        return true;
    }

    void pickPhysicalDevice()
    {
        uint32_t deviceCount = 0;
        vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
        if (deviceCount == 0) 
        {
            throw std::runtime_error("failed to find GPUs with Vulkan support!");
        }

        std::vector<VkPhysicalDevice> devices(deviceCount);
        vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

        for (const auto& device : devices) 
        {
            if (isDeviceSuitable(device)) 
            {
                physicalDevice = device;
                break;
            }
        }

        if (physicalDevice == VK_NULL_HANDLE) 
        {
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }
    bool isDeviceSuitable(VkPhysicalDevice device)
    {
        VulkanHelperFunctions::QueueFamilyIndices indices = VulkanHelperFunctions::findQueueFamilies(device, surface);

        bool extensionsSupported = checkDeviceExtensionSupport(device);
        VkPhysicalDeviceFeatures supportedFeatures;
        vkGetPhysicalDeviceFeatures(device, &supportedFeatures);

        return indices.isComplete() && extensionsSupported && supportedFeatures.samplerAnisotropy;
    }


    bool checkDeviceExtensionSupport(VkPhysicalDevice device) 
    {
        uint32_t extensionCount;
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);

        std::vector<VkExtensionProperties> availableExtensions(extensionCount);
        vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, availableExtensions.data());

        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }

        return requiredExtensions.empty();
    }

    void createLogicalDevice()
    {
        VulkanHelperFunctions::QueueFamilyIndices indices = VulkanHelperFunctions::findQueueFamilies(physicalDevice, surface);

        std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) 
        {
            VkDeviceQueueCreateInfo queueCreateInfo{};
            queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
            queueCreateInfo.queueFamilyIndex = queueFamily;
            queueCreateInfo.queueCount = 1;
            queueCreateInfo.pQueuePriorities = &queuePriority;
            queueCreateInfos.push_back(queueCreateInfo);
        }

        VkDeviceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
        createInfo.pQueueCreateInfos = queueCreateInfos.data();

        VkPhysicalDeviceFeatures deviceFeatures{};
        deviceFeatures.samplerAnisotropy = VK_TRUE;
        createInfo.pEnabledFeatures = &deviceFeatures;

        createInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
        createInfo.ppEnabledExtensionNames = deviceExtensions.data();

        if (enableValidationLayers) 
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(validationLayers.size());
            createInfo.ppEnabledLayerNames = validationLayers.data();
        }
        else 
        {
            createInfo.enabledLayerCount = 0;
        }

        if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create logical device!");
        }

        vkGetDeviceQueue(device, indices.graphicsFamily.value(), 0, &graphicsQueue);
        vkGetDeviceQueue(device, indices.presentFamily.value(), 0, &presentQueue);
    }

    void createSurface() 
    {
        if (glfwCreateWindowSurface(instance, window, nullptr, &surface) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create window surface!");
        }
    }

    SwapChainSupportDetails querySwapChainSupport(VkPhysicalDevice device) 
    {
        SwapChainSupportDetails details;
        vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface, &details.capabilities); uint32_t formatCount;
        vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, nullptr);

        if (formatCount != 0) 
        {
            details.formats.resize(formatCount);
            vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface, &formatCount, details.formats.data());
        }
        uint32_t presentModeCount;
        vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, nullptr);

        if (presentModeCount != 0) 
        {
            details.presentModes.resize(presentModeCount);
            vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &presentModeCount, details.presentModes.data());
        }
        return details;
    }

    VkSurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) 
    {
        for (const auto& availableFormat : availableFormats) 
        {
            if (availableFormat.format == VK_FORMAT_B8G8R8A8_SRGB && availableFormat.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) 
            {
                return availableFormat;
            }
        }

        return availableFormats[0];
    }

    void createSwapChain()
    {
        SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

        VkSurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        VkPresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        VkExtent2D extent = chooseSwapExtent(swapChainSupport.capabilities);
        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount)
        {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }
        VkSwapchainCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

        VulkanHelperFunctions::QueueFamilyIndices indices = VulkanHelperFunctions::findQueueFamilies(physicalDevice, surface);
        uint32_t queueFamilyIndices[] = { indices.graphicsFamily.value(), indices.presentFamily.value() };

        if (indices.graphicsFamily != indices.presentFamily) 
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        }
        else 
        {
            createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
            createInfo.queueFamilyIndexCount = 0; // Optional
            createInfo.pQueueFamilyIndices = nullptr; // Optional
        }

        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;

        if (vkCreateSwapchainKHR(device, &createInfo, nullptr, &swapChain) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create swap chain!");
        }

        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, nullptr);
        swapChainImages.resize(imageCount);
        vkGetSwapchainImagesKHR(device, swapChain, &imageCount, swapChainImages.data());
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    VkPresentModeKHR chooseSwapPresentMode(const std::vector<VkPresentModeKHR>& availablePresentModes) 
    {
        return VK_PRESENT_MODE_FIFO_KHR;
    }

    VkExtent2D chooseSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities) 
    {
        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        VkExtent2D extent;
        extent.width = width;
        extent.height = height;
        return extent;
    }

    void createImageViews()
    {
        swapChainImageViews.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) 
        {
            swapChainImageViews[i] = textureImageView = VulkanHelperFunctions::createImageView(device, swapChainImages[i], swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
        }
    }

    void createGraphicsPipeline()
    {
        auto vertShaderCode = readFile("./shaders/basicShaderV.spv");
        auto fragShaderCode = readFile("./shaders/basicShaderF.spv");

        VkShaderModule vertShaderModule = createShaderModule(vertShaderCode);
        VkShaderModule fragShaderModule = createShaderModule(fragShaderCode);

        VkPipelineShaderStageCreateInfo vertShaderStageInfo{};
        vertShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfo.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfo.module = vertShaderModule;
        vertShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo fragShaderStageInfo{};
        fragShaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfo.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfo.module = fragShaderModule;
        fragShaderStageInfo.pName = "main";
        VkPipelineShaderStageCreateInfo shaderStages[] = { vertShaderStageInfo, fragShaderStageInfo };

        //second pass
        auto vertShaderCodeSec = readFile("./shaders/secondPassShaderV.spv");
        auto fragShaderCodeSec = readFile("./shaders/secondPassShaderF.spv");

        VkShaderModule vertShaderModuleSecondPass = createShaderModule(vertShaderCodeSec);
        VkShaderModule fragShaderModuleSecondPass = createShaderModule(fragShaderCodeSec);

        VkPipelineShaderStageCreateInfo vertShaderStageInfoSecondPass{};
        vertShaderStageInfoSecondPass.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        vertShaderStageInfoSecondPass.stage = VK_SHADER_STAGE_VERTEX_BIT;
        vertShaderStageInfoSecondPass.module = vertShaderModuleSecondPass;
        vertShaderStageInfoSecondPass.pName = "main";
        VkPipelineShaderStageCreateInfo fragShaderStageInfoSecondPass{};
        fragShaderStageInfoSecondPass.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        fragShaderStageInfoSecondPass.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
        fragShaderStageInfoSecondPass.module = fragShaderModuleSecondPass;
        fragShaderStageInfoSecondPass.pName = "main";
        VkPipelineShaderStageCreateInfo shaderStagesSecondPass[] = { vertShaderStageInfoSecondPass, fragShaderStageInfoSecondPass };

        VkPipelineVertexInputStateCreateInfo vertexInputInfo{};
        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();
        vertexInputInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
        vertexInputInfo.vertexBindingDescriptionCount = 1;
        vertexInputInfo.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size());
        vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
        vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

        VkPipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
        inputAssembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkViewport viewport{};
        viewport.x = 0.0f;
        viewport.y = 0.0f;
        viewport.width = (float)swapChainExtent.width;
        viewport.height = (float)swapChainExtent.height;
        viewport.minDepth = 0.0f;
        viewport.maxDepth = 1.0f;
        inputAssembly.primitiveRestartEnable = VK_FALSE;

        VkRect2D scissor{};
        scissor.offset = { 0, 0 };
        scissor.extent = swapChainExtent;

        VkPipelineViewportStateCreateInfo viewportState{};
        viewportState.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
        viewportState.viewportCount = 1;
        viewportState.pViewports = &viewport;
        viewportState.scissorCount = 1;
        viewportState.pScissors = &scissor;

        VkPipelineRasterizationStateCreateInfo rasterizer{};
        rasterizer.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
        rasterizer.depthClampEnable = VK_FALSE;
        rasterizer.rasterizerDiscardEnable = VK_FALSE;
        rasterizer.polygonMode = VK_POLYGON_MODE_FILL;
        rasterizer.lineWidth = 1.0f;
        rasterizer.cullMode = VK_CULL_MODE_BACK_BIT;
        rasterizer.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterizer.depthBiasEnable = VK_FALSE;
        rasterizer.depthBiasConstantFactor = 0.0f; // Optional
        rasterizer.depthBiasClamp = 0.0f; // Optional
        rasterizer.depthBiasSlopeFactor = 0.0f; // Optional

        VkPipelineMultisampleStateCreateInfo multisampling{};
        multisampling.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
        multisampling.sampleShadingEnable = VK_FALSE;
        multisampling.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
        multisampling.minSampleShading = 1.0f; // Optional
        multisampling.pSampleMask = nullptr; // Optional
        multisampling.alphaToCoverageEnable = VK_FALSE; // Optional
        multisampling.alphaToOneEnable = VK_FALSE; // Optional

        VkPipelineColorBlendAttachmentState colorBlendAttachment{};
        colorBlendAttachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT | VK_COLOR_COMPONENT_A_BIT;
        colorBlendAttachment.blendEnable = VK_FALSE;
        colorBlendAttachment.srcColorBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstColorBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.colorBlendOp = VK_BLEND_OP_ADD; // Optional
        colorBlendAttachment.srcAlphaBlendFactor = VK_BLEND_FACTOR_ONE; // Optional
        colorBlendAttachment.dstAlphaBlendFactor = VK_BLEND_FACTOR_ZERO; // Optional
        colorBlendAttachment.alphaBlendOp = VK_BLEND_OP_ADD; // Optional

        VkPipelineColorBlendStateCreateInfo colorBlending{};
        colorBlending.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
        colorBlending.logicOpEnable = VK_FALSE;
        colorBlending.logicOp = VK_LOGIC_OP_COPY; // Optional
        colorBlending.attachmentCount = 1;
        colorBlending.pAttachments = &colorBlendAttachment;
        colorBlending.blendConstants[0] = 0.0f; // Optional
        colorBlending.blendConstants[1] = 0.0f; // Optional
        colorBlending.blendConstants[2] = 0.0f; // Optional
        colorBlending.blendConstants[3] = 0.0f; // Optional

        VkPipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = VK_COMPARE_OP_LESS;
        depthStencil.depthBoundsTestEnable = VK_FALSE;
        depthStencil.stencilTestEnable = VK_FALSE;

        VkPushConstantRange pushRange;
        pushRange.size = sizeof(PushConstantInfo);
        pushRange.offset = 0;
        pushRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout;
        pipelineLayoutInfo.pushConstantRangeCount = 1; // Optional
        pipelineLayoutInfo.pPushConstantRanges = &pushRange; // Optional
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create pipeline layout!");
        }
        
        depthStencil.depthTestEnable = VK_FALSE;
        rasterizer.cullMode = VK_CULL_MODE_NONE;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayoutSecondPass;
        if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayoutSecondPass) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create pipeline layout!");
        }

        VkGraphicsPipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
        pipelineInfo.stageCount = 2;
        pipelineInfo.pStages = shaderStages;
        pipelineInfo.pVertexInputState = &vertexInputInfo;
        pipelineInfo.pInputAssemblyState = &inputAssembly;
        pipelineInfo.pViewportState = &viewportState;
        pipelineInfo.pRasterizationState = &rasterizer;
        pipelineInfo.pMultisampleState = &multisampling;
        pipelineInfo.pDepthStencilState = &depthStencil;
        pipelineInfo.pColorBlendState = &colorBlending;
        pipelineInfo.pDynamicState = nullptr; // Optional
        pipelineInfo.layout = pipelineLayout;
        pipelineInfo.renderPass = renderPass;
        pipelineInfo.subpass = 0;
        pipelineInfo.basePipelineHandle = VK_NULL_HANDLE; // Optional
        pipelineInfo.basePipelineIndex = -1; // Optional

        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipeline) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        pipelineInfo.pStages = shaderStagesSecondPass;
        pipelineInfo.layout = pipelineLayoutSecondPass;
        pipelineInfo.subpass = 1;
        if (vkCreateGraphicsPipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &graphicsPipelineSecondPass) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create graphics pipeline!");
        }

        vkDestroyShaderModule(device, vertShaderModule, nullptr);
        vkDestroyShaderModule(device, fragShaderModule, nullptr);
        vkDestroyShaderModule(device, vertShaderModuleSecondPass, nullptr);
        vkDestroyShaderModule(device, fragShaderModuleSecondPass, nullptr);
    }

    static std::vector<char> readFile(const std::string& fileLoc)
    {
        std::ifstream file(fileLoc, std::ios::ate | std::ios::binary);

        if (!file.is_open()) 
        {
            throw std::runtime_error("failed to open file!");
        }

        size_t fileSize = (size_t)file.tellg();
        std::vector<char> buffer(fileSize);
        file.seekg(0);
        file.read(buffer.data(), fileSize);
        file.close();

        return buffer;
    }

    VkShaderModule createShaderModule(const std::vector<char>& code)
    {
        VkShaderModuleCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        createInfo.codeSize = code.size();
        createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkShaderModule shaderModule;
        if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create shader module!");
        }
        return shaderModule;
    }

    void createRenderPass()
    {
        VkAttachmentDescription colorAttachment{};
        colorAttachment.format = swapChainImageFormat;
        colorAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        colorAttachment.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription depthAttachment{};
        depthAttachment.format = VK_FORMAT_D32_SFLOAT;
        depthAttachment.samples = VK_SAMPLE_COUNT_1_BIT;
        depthAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        depthAttachment.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        depthAttachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        depthAttachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        depthAttachment.finalLayout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentDescription colorAttachment2{};
        colorAttachment2.format = swapChainImageFormat;
        colorAttachment2.samples = VK_SAMPLE_COUNT_1_BIT;
        colorAttachment2.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        colorAttachment2.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        colorAttachment2.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
        colorAttachment2.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
        colorAttachment2.initialLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        colorAttachment2.finalLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorAttachmentRef{};
        colorAttachmentRef.attachment = 2;
        colorAttachmentRef.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkAttachmentReference depthAttachmentRef{};
        depthAttachmentRef.attachment = 1;
        depthAttachmentRef.layout = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

        VkAttachmentReference colorInputReference{};
        colorInputReference.attachment = 2;
        colorInputReference.layout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;

        VkAttachmentReference colorOutputReference{};
        colorOutputReference.attachment = 0;
        colorOutputReference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

        VkSubpassDescription subpass{};
        subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpass.colorAttachmentCount = 1;
        subpass.pColorAttachments = &colorAttachmentRef;
        subpass.pDepthStencilAttachment = &depthAttachmentRef;

        VkSubpassDescription subpassFinal{};
        subpassFinal.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
        subpassFinal.inputAttachmentCount = 1;
        subpassFinal.pInputAttachments = &colorInputReference;
        subpassFinal.colorAttachmentCount = 1;
        subpassFinal.pColorAttachments = &colorOutputReference;
        subpassFinal.pDepthStencilAttachment = nullptr;

        std::array<VkSubpassDescription, 2> subpasses{subpass, subpassFinal};

        VkSubpassDependency dependency{};
        dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
        dependency.dstSubpass = 0;
        dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.srcAccessMask = 0;
        dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

        VkSubpassDependency dependency2{};
        dependency2.srcSubpass = 0;
        dependency2.dstSubpass = 1;
        dependency2.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency2.srcAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        dependency2.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
        dependency2.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        
        std::array<VkSubpassDependency, 2> dependencies = { dependency, dependency2 };
        
        std::array<VkAttachmentDescription, 3> attachments = { colorAttachment, depthAttachment, colorAttachment2 };
        VkRenderPassCreateInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
        renderPassInfo.attachmentCount = attachments.size();
        renderPassInfo.pAttachments = attachments.data();
        renderPassInfo.subpassCount = subpasses.size();
        renderPassInfo.pSubpasses = subpasses.data();
        renderPassInfo.dependencyCount = dependencies.size();
        renderPassInfo.pDependencies = dependencies.data();

        if (vkCreateRenderPass(device, &renderPassInfo, nullptr, &renderPass) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create render pass!");
        }
    }

    void createFramebuffers() 
    {
        swapChainFramebuffers.resize(swapChainImageViews.size());
        imguiFramebuffers.resize(swapChainImageViews.size());

        for (size_t i = 0; i < swapChainImageViews.size(); i++) 
        {
            //ImGUI main framebuffer
            std::array<VkImageView, 3> attachments = 
            {
                swapChainImageViews[i],
                depthImageView,
                colorImageView
            };

            VkFramebufferCreateInfo framebufferInfo{};
            framebufferInfo.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo.renderPass = renderPass;
            framebufferInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
            framebufferInfo.pAttachments = attachments.data();
            framebufferInfo.width = swapChainExtent.width;
            framebufferInfo.height = swapChainExtent.height;
            framebufferInfo.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo, nullptr, &swapChainFramebuffers[i]) != VK_SUCCESS) 
            {
                throw std::runtime_error("failed to create framebuffer!");
            }


            //ImGUI framebuffer
            std::array<VkImageView, 1> attachments2 =
            {
                swapChainImageViews[i]
            };

            VkFramebufferCreateInfo framebufferInfo2{};
            framebufferInfo2.sType = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
            framebufferInfo2.renderPass = imguiRenderPass;
            framebufferInfo2.attachmentCount = static_cast<uint32_t>(attachments2.size());
            framebufferInfo2.pAttachments = attachments2.data();
            framebufferInfo2.width = swapChainExtent.width;
            framebufferInfo2.height = swapChainExtent.height;
            framebufferInfo2.layers = 1;

            if (vkCreateFramebuffer(device, &framebufferInfo2, nullptr, &imguiFramebuffers[i]) != VK_SUCCESS)
            {
                throw std::runtime_error("failed to create framebuffer!");
            }
        }
    }

    void createCommandBuffers()
    {
        commandBuffers.resize(MAX_FRAMES_IN_FLIGHT);

        VkCommandBufferAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        allocInfo.commandPool = commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = (uint32_t)commandBuffers.size();

        if (vkAllocateCommandBuffers(device, &allocInfo, commandBuffers.data()) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to allocate command buffers!");
        }
    }

    void createSyncObjects() 
    {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

        VkSemaphoreCreateInfo semaphoreInfo{};
        semaphoreInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            if (vkCreateSemaphore(device, &semaphoreInfo, nullptr, &imageAvailableSemaphores[i]) != VK_SUCCESS ||
                vkCreateSemaphore(device, &semaphoreInfo, nullptr, &renderFinishedSemaphores[i]) != VK_SUCCESS ||
                vkCreateFence(device, &fenceInfo, nullptr, &inFlightFences[i]) != VK_SUCCESS) 
            {

                throw std::runtime_error("failed to create synchronization objects for a frame!");
            }
        }
    }

    void createVertexBuffer()
    {
        std::vector<VkBuffer> vertexBufferTemp;
        std::vector<VkDeviceMemory>  vertexBufferMemoryTemp;
        vertexBufferTemp.resize(MAX_FRAMES_IN_FLIGHT);
        vertexBufferMemoryTemp.resize(MAX_FRAMES_IN_FLIGHT);
        vertexBuffer.resize(MAX_FRAMES_IN_FLIGHT);
        vertexBufferMemory.resize(MAX_FRAMES_IN_FLIGHT);
        vertexBufferSecondPass.resize(MAX_FRAMES_IN_FLIGHT);
        vertexBufferMemorySecondPass.resize(MAX_FRAMES_IN_FLIGHT);
        vertexBufferLightSource.resize(MAX_FRAMES_IN_FLIGHT);
        vertexBufferMemoryLightSource.resize(MAX_FRAMES_IN_FLIGHT);

        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;

        //50 MB
        VkDeviceSize bufferSize = 500000;

        /*VBO 1*/
        for (int i = 0; i < vertexBufferTemp.size(); i++)
        {
            VulkanHelperFunctions::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, vertexBufferTemp[i], vertexBufferMemoryTemp[i]);

            void* data;
            vkMapMemory(device, vertexBufferMemoryTemp[i], 0, sizeof(Vertex), 0, &data);
            memcpy(data, vertices.data(), sizeof(Vertex) * vertices.size());
            vkUnmapMemory(device, vertexBufferMemoryTemp[i]);
        }

        for (int i = 0; i < vertexBuffer.size(); i++)
        {
            VulkanHelperFunctions::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBuffer[i], vertexBufferMemory[i]);
            copyBuffer(vertexBufferTemp[i], vertexBuffer[i], bufferSize);
        }

        /*VBO 2 Second Pass*/
        for (int i = 0; i < vertexBufferTemp.size(); i++)
        {
            void* data;
            vkMapMemory(device, vertexBufferMemoryTemp[i], 0, sizeof(Vertex), 0, &data);
            memcpy(data, verticesSecondPass.data(), sizeof(Vertex) * verticesSecondPass.size());
            vkUnmapMemory(device, vertexBufferMemoryTemp[i]);
        }

        for (int i = 0; i < vertexBufferSecondPass.size(); i++)
        {
            VulkanHelperFunctions::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, vertexBufferSecondPass[i], vertexBufferMemorySecondPass[i]);
            copyBuffer(vertexBufferTemp[i], vertexBufferSecondPass[i], bufferSize);
            vkFreeMemory(device, vertexBufferMemoryTemp[i], nullptr);
            vkDestroyBuffer(device, vertexBufferTemp[i], nullptr);
        }
    }

    void createDescriptorPool()
    {
        VkDescriptorPoolSize poolSizeImgui[] =
        {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000 }
        };

        VkDescriptorPoolCreateInfo poolInfoImgui{};
        poolInfoImgui.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfoImgui.poolSizeCount = 10;
        poolInfoImgui.pPoolSizes = poolSizeImgui;
        poolInfoImgui.maxSets = 1000;
        
        if (vkCreateDescriptorPool(device, &poolInfoImgui, nullptr, &imguiDescriptorPool) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to create descriptor pool!");
        }

        VkDescriptorPoolSize poolSize[11] =
        {
            { VK_DESCRIPTOR_TYPE_SAMPLER, 10 },
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 10 },
            { VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 10 },
            { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 10 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 10 },
            { VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 10 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 10 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 10 },
            { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 10 },
            { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 10 },
            { VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 10 }
        };

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 11;
        poolInfo.pPoolSizes = poolSize;
        poolInfo.maxSets = 100;

        if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &descriptorPool) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor pool!");
        }

    }
    
    void createDescriptorSetLayout()
    {
        VkDescriptorSetLayoutBinding uboLayoutBinding{};
        uboLayoutBinding.binding = 0;
        uboLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        uboLayoutBinding.descriptorCount = 1;
        uboLayoutBinding.stageFlags = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
        uboLayoutBinding.pImmutableSamplers = nullptr; // Optional

        VkDescriptorSetLayoutBinding samplerLayoutBinding{};
        samplerLayoutBinding.binding = 1;
        samplerLayoutBinding.descriptorCount = 1;
        samplerLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        samplerLayoutBinding.pImmutableSamplers = nullptr;
        samplerLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        VkDescriptorSetLayoutBinding lightingVariablesBinding{};
        lightingVariablesBinding.binding = 2;
        lightingVariablesBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        lightingVariablesBinding.descriptorCount = 1;
        lightingVariablesBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;
        lightingVariablesBinding.pImmutableSamplers = nullptr; // Optional

        VkDescriptorSetLayoutBinding inputAttachmentLayoutBinding{};
        inputAttachmentLayoutBinding.binding = 0;
        inputAttachmentLayoutBinding.descriptorCount = 1;
        inputAttachmentLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
        inputAttachmentLayoutBinding.pImmutableSamplers = nullptr;
        inputAttachmentLayoutBinding.stageFlags = VK_SHADER_STAGE_FRAGMENT_BIT;

        std::array<VkDescriptorSetLayoutBinding, 3> firstPassBindings = { uboLayoutBinding, samplerLayoutBinding, lightingVariablesBinding };
        VkDescriptorSetLayoutCreateInfo layoutInfoFirstPass{};
        layoutInfoFirstPass.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfoFirstPass.bindingCount = static_cast<uint32_t>(firstPassBindings.size() );
        layoutInfoFirstPass.pBindings = firstPassBindings.data();

        if (vkCreateDescriptorSetLayout(device, &layoutInfoFirstPass, nullptr, &descriptorSetLayout) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor set layout!");
        }

        std::array<VkDescriptorSetLayoutBinding, 1> secondPassBindings = { inputAttachmentLayoutBinding };
        VkDescriptorSetLayoutCreateInfo layoutInfoSecondPass{};
        layoutInfoSecondPass.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfoSecondPass.bindingCount = static_cast<uint32_t>(secondPassBindings.size());
        layoutInfoSecondPass.pBindings = secondPassBindings.data();
        if (vkCreateDescriptorSetLayout(device, &layoutInfoSecondPass, nullptr, &descriptorSetLayoutSecondPass) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create descriptor set layout!");
        }
    }

    void createDescriptorSets()
    {
        std::vector<VkDescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo.pSetLayouts = layouts.data();

        descriptorSets.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo, descriptorSets.data()) != VK_SUCCESS) 
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) 
        {
            VkDescriptorBufferInfo bufferInfo{};
            bufferInfo.buffer = uniformBuffers[i];
            bufferInfo.offset = 0;
            bufferInfo.range = sizeof(UniformBufferObject);

            VkDescriptorBufferInfo bufferInfoShaderInfo{};
            bufferInfoShaderInfo.buffer = uniformBuffersShaderInfo[i];
            bufferInfoShaderInfo.offset = 0;
            bufferInfoShaderInfo.range = sizeof(UniformShaderInfo);

            VkDescriptorImageInfo imageInfo{};
            imageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            imageInfo.imageView = textureImageView;
            imageInfo.sampler = textureSampler;

            std::array<VkWriteDescriptorSet, 3> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSets[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pBufferInfo = &bufferInfo;

            descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[1].dstSet = descriptorSets[i];
            descriptorWrites[1].dstBinding = 1;
            descriptorWrites[1].dstArrayElement = 0;
            descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
            descriptorWrites[1].descriptorCount = 1;
            descriptorWrites[1].pImageInfo = &imageInfo;

            descriptorWrites[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[2].dstSet = descriptorSets[i];
            descriptorWrites[2].dstBinding = 2;
            descriptorWrites[2].dstArrayElement = 0;
            descriptorWrites[2].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            descriptorWrites[2].descriptorCount = 1;
            descriptorWrites[2].pBufferInfo = &bufferInfoShaderInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }

        std::vector<VkDescriptorSetLayout> layouts2(MAX_FRAMES_IN_FLIGHT, descriptorSetLayoutSecondPass);
        VkDescriptorSetAllocateInfo allocInfo2{};
        allocInfo2.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo2.descriptorPool = descriptorPool;
        allocInfo2.descriptorSetCount = static_cast<uint32_t>(MAX_FRAMES_IN_FLIGHT);
        allocInfo2.pSetLayouts = layouts2.data();

        descriptorSetsSecondPass.resize(MAX_FRAMES_IN_FLIGHT);
        if (vkAllocateDescriptorSets(device, &allocInfo2, descriptorSetsSecondPass.data()) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to allocate descriptor sets!");
        }

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            VkDescriptorImageInfo colorImageInfo{};
            colorImageInfo.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            colorImageInfo.imageView = colorImageView;
            colorImageInfo.sampler = VK_NULL_HANDLE;

            std::array<VkWriteDescriptorSet, 1> descriptorWrites{};

            descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptorWrites[0].dstSet = descriptorSetsSecondPass[i];
            descriptorWrites[0].dstBinding = 0;
            descriptorWrites[0].dstArrayElement = 0;
            descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT;
            descriptorWrites[0].descriptorCount = 1;
            descriptorWrites[0].pImageInfo = &colorImageInfo;

            vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()), descriptorWrites.data(), 0, nullptr);
        }
    }

    void createIndexBuffer()
    {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        VkBuffer stagingBuffer;
        VkDeviceMemory stagingBufferMemory;
        VulkanHelperFunctions::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, stagingBuffer, stagingBufferMemory);
        VulkanHelperFunctions::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBuffer, indexBufferMemory);
        VulkanHelperFunctions::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, indexBufferSecondPass, indexBufferMemorySecondPass);

        void* data;
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indices.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        bufferSize = sizeof(indicesSecondPass[0]) * indicesSecondPass.size();
        vkMapMemory(device, stagingBufferMemory, 0, bufferSize, 0, &data);
        memcpy(data, indicesSecondPass.data(), (size_t)bufferSize);
        vkUnmapMemory(device, stagingBufferMemory);
        copyBuffer(stagingBuffer, indexBufferSecondPass, bufferSize);

        vkDestroyBuffer(device, stagingBuffer, nullptr);
        vkFreeMemory(device, stagingBufferMemory, nullptr);
    }

    void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, VkDeviceSize size) 
    {
        VkCommandBuffer commandBuffer = VulkanHelperFunctions::beginSingleTimeCommands(device, commandPool);
        VkBufferCopy copyRegion{};
        copyRegion.size = size;
        vkCmdCopyBuffer(commandBuffer, srcBuffer, dstBuffer, 1, &copyRegion);
        VkBufferMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.buffer = dstBuffer;
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.offset = 0;
        barrier.size = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(commandBuffer, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, 0,
            0, nullptr,
            1, &barrier,
            0, nullptr);
        VulkanHelperFunctions::endSingleTimeCommands(device, commandPool, graphicsQueue, commandBuffer);
    }

    void createUniformBuffers()
    {
        uniformBuffers.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersMemory.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersShaderInfo.resize(MAX_FRAMES_IN_FLIGHT);
        uniformBuffersShaderInfoMemory.resize(MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
        {
            VkDeviceSize bufferSize = sizeof(UniformBufferObject);
            VulkanHelperFunctions::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffers[i], uniformBuffersMemory[i]);
            bufferSize = sizeof(UniformShaderInfo);
            VulkanHelperFunctions::createBuffer(device, physicalDevice, bufferSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, uniformBuffersShaderInfo[i], uniformBuffersShaderInfoMemory[i]);
        }
    }
    
    void createTextureImageView() 
    {
        textureImageView = VulkanHelperFunctions::createImageView(device, textureImage, VK_FORMAT_R8G8B8A8_SRGB, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    void createDepthResources()
    {
        VkFormat depthFormat = VK_FORMAT_D32_SFLOAT;
        VulkanHelperFunctions::createImage(physicalDevice, device, swapChainExtent.width, swapChainExtent.height, depthFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, depthImage, depthImageMemory);
        depthImageView = textureImageView = VulkanHelperFunctions::createImageView(device, depthImage, depthFormat, VK_IMAGE_ASPECT_DEPTH_BIT);

        VulkanHelperFunctions::createImage(physicalDevice, device, swapChainExtent.width, swapChainExtent.height, swapChainImageFormat, VK_IMAGE_TILING_OPTIMAL, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_INPUT_ATTACHMENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, colorImage, colorImageMemory);
        VulkanHelperFunctions::transitionImageLayout(device, commandPool, graphicsQueue, colorImage, swapChainImageFormat, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        colorImageView = textureImageView = VulkanHelperFunctions::createImageView(device, colorImage, swapChainImageFormat, VK_IMAGE_ASPECT_COLOR_BIT);
    }

    void loadModel()
    {
        tinyobj::attrib_t attrib;
        std::vector<tinyobj::shape_t> shapes;
        std::vector<tinyobj::material_t> materials;
        std::string warn, err;

        if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, MODEL_PATH.c_str())) 
        {
            throw std::runtime_error(warn + err);
        }
        for (const auto& shape : shapes)
        {
            for (const auto& index : shape.mesh.indices) 
            {
                auto size = shape.mesh.material_ids.size();
                Vertex vertex{};
                vertex.pos = 
                {
                    attrib.vertices[3 * static_cast<double>(index.vertex_index) + 0],
                    attrib.vertices[3 * static_cast<double>(index.vertex_index) + 1],
                    attrib.vertices[3 * static_cast<double>(index.vertex_index) + 2],
                };

                vertex.texCoord = 
                {
                           attrib.texcoords[2 * static_cast<double>(index.texcoord_index) + 0],
                    1.0f - attrib.texcoords[2 * static_cast<double>(index.texcoord_index) + 1]
                };
                vertex.normal = 
                {
                    attrib.normals[3 * static_cast<double>(index.normal_index) + 0],
                    attrib.normals[3 * static_cast<double>(index.normal_index) + 1],
                    attrib.normals[3 * static_cast<double>(index.normal_index) + 2],
                };
                vertex.color = { 1.0f, 1.0f, 1.0f };

                if (uniqueVertices.count(vertex) == 0) 
                {
                    uniqueVertices[vertex] = static_cast<uint32_t>(vertices.size());
                    vertices.push_back(vertex);
                }
                indices.push_back(uniqueVertices[vertex]);
            }
        }
    }

    void mainLoop() {
        while (!glfwWindowShouldClose(window)) 
        {
            glfwPollEvents();        
            drawFrame();
            updateMovement();
        }
        vkDeviceWaitIdle(device);
    }

    void updateMovement()
    {

    }

    void drawFrame()
    {
        static bool drawWithDiffVBO = false;
        static double savedTime = 0;

        //initalize camera
        static Camera camera(
            glm::vec3(1.0, 1.0, 0.0),
            glm::vec3(.0, .0, .0),
            glm::vec3(.0, 1.0, 0.0),
            WIDTH,
            HEIGHT
        );

        //ImGui Variables
        static bool firstDraw = true;
        static float lightPos[3];
        static float cameraPos[3];
        static bool rotateObject;

        //initalize IMGUI variables
        if (firstDraw)
        {
            memset(lightPos, 0, sizeof(lightPos));
            lightPos[2] = 1.8f;
            cameraPos[0] = cameraPos[1] = cameraPos[1] = 2.0f;
            rotateObject = false;
            firstDraw = false;
        }

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        ImVec4 color(1.0, 1.0, 0.0, 0.7);
        ImGui::Begin("Dear ImGui Style Editor");
            ImGui::TextColored(color, "Current vertex count: %d", vertices.size());
            ImGui::TextColored(color, "Current indices count: %d", indices.size());
            ImGui::SliderFloat3("x,y,z for lightPosition", lightPos, 0.0, 10.0);
            ImGui::SliderFloat3("x,y,z for Camera Position", cameraPos, 0.0, 10.0);
            ImVec2 buttonSize = {150, 30};
            if (ImGui::Button("Rotate Object", buttonSize))
            {
                rotateObject = !rotateObject;
            }
        ImGui::End();
        ImGui::Render();

        vkWaitForFences(device, 1, &inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        VkResult result = vkAcquireNextImageKHR(device, swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if( VK_ERROR_OUT_OF_DATE_KHR == result )
        {
            recreateSwapChain();
            return;
        }

        // Check if a previous frame is using this image (i.e. there is its fence to wait on)
        if (imagesInFlight[imageIndex] != VK_NULL_HANDLE) 
        {
            vkWaitForFences(device, 1, &imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }

        // Mark the image as now being in use by this frame
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        UniformShaderInfo shaderInfo{};
        shaderInfo.lightColor = glm::vec3(0.8, .8, 0.8);
        shaderInfo.lightPos = glm::vec3(lightPos[0], lightPos[1], lightPos[2]);
        auto glmCameraPos = glm::vec3(cameraPos[0], cameraPos[1], cameraPos[2]);
        camera.SetCurrentPosition(glmCameraPos);
        updateUniformBuffer(currentFrame, shaderInfo, camera);
        startRender(commandBuffers[currentFrame]);
            render(commandBuffers[currentFrame], renderPass, swapChainFramebuffers[imageIndex], vertexBuffer[currentFrame], descriptorSets[currentFrame], graphicsPipeline, pipelineLayout, descriptorSetsSecondPass[currentFrame], vertexBufferSecondPass[currentFrame], shaderInfo.lightPos, rotateObject);
            renderImgui(commandBuffers[currentFrame], imguiRenderPass, imguiFramebuffers[imageIndex]);
        finishRender(commandBuffers[currentFrame]);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        VkSemaphore waitSemaphores[] = { imageAvailableSemaphores[currentFrame] };
        VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = waitSemaphores;
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffers[currentFrame];
        VkSemaphore signalSemaphores[] = { renderFinishedSemaphores[currentFrame] };
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = signalSemaphores;

        vkResetFences(device, 1, &inFlightFences[currentFrame]);
        if (vkQueueSubmit(graphicsQueue, 1, &submitInfo, inFlightFences[currentFrame]) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to submit draw command buffer!");
        }

        VkPresentInfoKHR presentInfo{};
        presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;

        presentInfo.waitSemaphoreCount = 1;
        presentInfo.pWaitSemaphores = signalSemaphores;
        VkSwapchainKHR swapChains[] = { swapChain };
        presentInfo.swapchainCount = 1;
        presentInfo.pSwapchains = swapChains;
        presentInfo.pImageIndices = &imageIndex;
        result = vkQueuePresentKHR(presentQueue, &presentInfo);
        if (result == VK_ERROR_OUT_OF_DATE_KHR || result == VK_SUBOPTIMAL_KHR)
        {
            recreateSwapChain();
        }
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void updateUniformBuffer(uint32_t currentFrame, const UniformShaderInfo& shaderInfo, const Camera& camera)
    {
        UniformBufferObject ubo{};
        ubo.view = camera.GetViewMatrix();
        ubo.proj = camera.GetPerspectiveMatrix();
        ubo.viewPos = camera.GetCurrentPosition();
        ubo.proj[1][1] *= -1;
        void* data;

        vkMapMemory(device, uniformBuffersMemory[currentFrame], 0, sizeof(ubo), 0, &data);
        memcpy(data, &ubo, sizeof(ubo));
        vkUnmapMemory(device, uniformBuffersMemory[currentFrame]);

        vkMapMemory(device, uniformBuffersShaderInfoMemory[currentFrame], 0, sizeof(shaderInfo), 0, &data);
        memcpy(data, &shaderInfo, sizeof(shaderInfo));
        vkUnmapMemory(device, uniformBuffersShaderInfoMemory[currentFrame]);
    }

    void startRender(VkCommandBuffer commandBuffer)
    {
        vkResetCommandBuffer(commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = 0; // Optional
        beginInfo.pInheritanceInfo = nullptr; // Optional

        if (vkBeginCommandBuffer(commandBuffer, &beginInfo) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to begin recording command buffer!");
        }
    }

    void render(VkCommandBuffer commandBuffer, VkRenderPass renderPass, VkFramebuffer mainRenderframebuffer, VkBuffer vertexBuffer, VkDescriptorSet descriptorSet, VkPipeline graphicsPipeline, VkPipelineLayout pipelineLayout, VkDescriptorSet descriptorSetSecondPass, VkBuffer vertexBufferSecondPass, glm::vec3 lightPos, bool rotateObject)
    {
        
        static auto startTime = std::chrono::high_resolution_clock::now();
        static auto currentTime = std::chrono::high_resolution_clock::now();
        if (rotateObject)
        {
            currentTime = std::chrono::high_resolution_clock::now();
        }
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        VkDeviceSize offsets[] = { 0 };
        std::array<VkClearValue, 3> clearValues{};
        clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
        clearValues[1].depthStencil = { 1.0f, 0 };
        clearValues[2].color = { 0.0f, 0.0f, 0.0f, 1.0f };

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = mainRenderframebuffer;
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;
        renderPassInfo.clearValueCount = static_cast<uint32_t>(clearValues.size());
        renderPassInfo.pClearValues = clearValues.data();

        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipeline);
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        PushConstantInfo pushConstantInfo{};
        pushConstantInfo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        pushConstantInfo.model = glm::rotate(pushConstantInfo.model, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantInfo), &pushConstantInfo);
        vkCmdDrawIndexed(commandBuffer, indices.size(), 1, 0, 0, 0);
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBuffer, 0, VK_INDEX_TYPE_UINT32);
        pushConstantInfo.model = glm::mat4(1.0f);
        pushConstantInfo.model = glm::translate(pushConstantInfo.model, lightPos);
        pushConstantInfo.model = glm::scale(pushConstantInfo.model, glm::vec3(0.2, 0.2, 1.0));
        vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstantInfo), &pushConstantInfo);
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBufferSecondPass, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBufferSecondPass, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(commandBuffer, indicesSecondPass.size(), 1, 0, 0, 0);

        vkCmdNextSubpass(commandBuffer, VK_SUBPASS_CONTENTS_INLINE);
        vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBufferSecondPass, offsets);
        vkCmdBindIndexBuffer(commandBuffer, indexBufferSecondPass, 0, VK_INDEX_TYPE_UINT32);
        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, graphicsPipelineSecondPass);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayoutSecondPass, 0, 1, &descriptorSetSecondPass, 0, NULL);
        vkCmdDrawIndexed(commandBuffer, indicesSecondPass.size(), 1, 0, 0, 0);

        vkCmdEndRenderPass(commandBuffer);
    }

    void renderImgui(VkCommandBuffer commandBuffer, VkRenderPass renderPass, VkFramebuffer framebuffer)
    {
        std::array<VkClearValue, 2> clearValues{};
        clearValues[0].color = { 0.0f, 0.0f, 0.0f, 1.0f };
        clearValues[1].depthStencil = { 1.0f, 0 };

        VkRenderPassBeginInfo renderPassInfo{};
        renderPassInfo.sType = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
        renderPassInfo.renderPass = renderPass;
        renderPassInfo.framebuffer = framebuffer;
        renderPassInfo.renderArea.offset = { 0, 0 };
        renderPassInfo.renderArea.extent = swapChainExtent;
        renderPassInfo.clearValueCount = clearValues.size();
        renderPassInfo.pClearValues = clearValues.data();
        vkCmdBeginRenderPass(commandBuffer, &renderPassInfo, VK_SUBPASS_CONTENTS_INLINE);
        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffer);
        vkCmdEndRenderPass(commandBuffer);
    }

    void finishRender(VkCommandBuffer commandBuffer)
    {
        if (vkEndCommandBuffer(commandBuffer) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to record command buffer!");
        }
    }

    void recreateSwapChain()
    {
        vkDeviceWaitIdle(device);
        cleanupSwapchain();
        //TODO in order to resize window
        //set depth image to new width and height
        //set framebuffers and imgui framebuffers
        //set pipeline
        //set pipeline layout
        //set swap chain image view
        // set swapchain
        createSwapChain();
        createImageViews();
        createDepthResources();
        createFramebuffers();
        createGraphicsPipeline();
    }

    void cleanupSwapchain()
    {
        vkDestroyImageView(device, depthImageView, nullptr);
        vkDestroyImage(device, depthImage, nullptr);
        vkFreeMemory(device, depthImageMemory, nullptr);

        for (auto framebuffer : swapChainFramebuffers)
        {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        swapChainFramebuffers.clear();

        for (auto framebuffer : imguiFramebuffers)
        {
            vkDestroyFramebuffer(device, framebuffer, nullptr);
        }
        imguiFramebuffers.clear();

        vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
        vkDestroyPipelineLayout(device, pipelineLayoutSecondPass, nullptr);
        vkDestroyPipeline(device, graphicsPipeline, nullptr);
        vkDestroyPipeline(device, graphicsPipelineSecondPass, nullptr);

        for (auto imageView : swapChainImageViews)
        {
            vkDestroyImageView(device, imageView, nullptr);
        }
        swapChainImageViews.clear();

        vkDestroySwapchainKHR(device, swapChain, nullptr);
    }

    void cleanup() 
    {
        cleanupSwapchain();

        //ImGui Cleanup
        vkDestroyRenderPass(device, imguiRenderPass, nullptr);
        vkDestroyDescriptorPool(device, imguiDescriptorPool, nullptr);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        vkDestroySampler(device, textureSampler, nullptr);
        vkDestroyImageView(device, textureImageView, nullptr);
        vkDestroyImage(device, textureImage, nullptr);
        vkFreeMemory(device, textureImageMemory, nullptr);
        vkDestroyImageView(device, colorImageView, nullptr);
        vkDestroyImage(device, colorImage, nullptr);
        vkFreeMemory(device, colorImageMemory, nullptr);

        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        
        for (int i = 0; i < renderFinishedSemaphores.size(); i++)
        {
            vkDestroySemaphore(device, renderFinishedSemaphores[i], nullptr);
            vkDestroySemaphore(device, imageAvailableSemaphores[i], nullptr);
            vkDestroyFence(device, inFlightFences[i], nullptr);
        }

        for (size_t i = 0; i < uniformBuffers.size(); i++)
        {
            vkDestroyBuffer(device, uniformBuffers[i], nullptr);
            vkFreeMemory(device, uniformBuffersMemory[i], nullptr);
        }
        
        for (size_t i = 0; i < uniformBuffersShaderInfo.size(); i++)
        {
            vkDestroyBuffer(device, uniformBuffersShaderInfo[i], nullptr);
            vkFreeMemory(device, uniformBuffersShaderInfoMemory[i], nullptr);
        }

        for (int i = 0; i < vertexBuffer.size(); i++)
        {
            vkDestroyBuffer(device, vertexBuffer[i], nullptr);
            vkFreeMemory(device, vertexBufferMemory[i], nullptr);
        }

        for (int i = 0; i < vertexBufferLightSource.size(); i++)
        {
            vkDestroyBuffer(device, vertexBufferLightSource[i], nullptr);
            vkFreeMemory(device, vertexBufferMemoryLightSource[i], nullptr);
        }

        for (int i = 0; i < vertexBufferSecondPass.size(); i++)
        {
            vkDestroyBuffer(device, vertexBufferSecondPass[i], nullptr);
            vkFreeMemory(device, vertexBufferMemorySecondPass[i], nullptr);
        }

        vkDestroyBuffer(device, indexBuffer, nullptr);
        vkFreeMemory(device, indexBufferMemory, nullptr);
        vkDestroyBuffer(device, indexBufferSecondPass, nullptr);
        vkFreeMemory(device, indexBufferMemorySecondPass, nullptr);

        vkDestroyRenderPass(device, renderPass, nullptr);

        vkDestroyCommandPool(device, commandPool, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptorSetLayoutSecondPass, nullptr);
        
        vkDestroySurfaceKHR(instance, surface, nullptr);
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);
        glfwTerminate();
    }
};

int main() {
    HelloTriangleApplication app {};

    try {
        app.run();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}