#pragma once

#include <filesystem>
#include <memory>

class Configuration
{
public:
    [[nodiscard]] static Configuration* getInstance();

    [[nodiscard]] static std::filesystem::path getShaderSourceDir();

    [[nodiscard]] static std::filesystem::path getShaderDir();

    [[nodiscard]] static std::filesystem::path getShaderFilePath(const std::string& shaderFile);

    [[nodiscard]] static std::filesystem::path getHairFilePath(const std::string& fileName);

private:
    Configuration();

    static std::unique_ptr<Configuration> sInstance;

    std::string mBaseDirPath       = "C:/Users/andromeda/CLionProjects/thesispromax";
    std::string mShadersDirName    = "Shader";
    std::string mShadersBinDirName = "Shader/bin";
    std::string mHairDirName       = "HairModels";
};
