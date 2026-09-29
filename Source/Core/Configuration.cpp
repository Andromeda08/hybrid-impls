#include "Configuration.hpp"

std::unique_ptr<Configuration> Configuration::sInstance = nullptr;

Configuration* Configuration::getInstance()
{
    if (!sInstance)
    {
        sInstance = std::unique_ptr<Configuration>(new Configuration());
    }
    return sInstance.get();
}

std::filesystem::path Configuration::getShaderSourceDir()
{
    const Configuration* config = getInstance();
    return std::format("{}/{}", config->mBaseDirPath, config->mShadersDirName);
}

std::filesystem::path Configuration::getShaderDir()
{
    const Configuration* config = getInstance();
    return std::format("{}/{}", config->mBaseDirPath, config->mShadersBinDirName);
}

std::filesystem::path Configuration::getShaderFilePath(const std::string& shaderFile)
{
    const Configuration* config = getInstance();
    return std::format("{}/{}/{}", config->mBaseDirPath, config->mShadersBinDirName, shaderFile);
}

Configuration::Configuration() = default;
