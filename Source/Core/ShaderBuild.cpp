#include "ShaderBuild.hpp"

#include <array>
#include <fstream>
#include <print>
#include <ranges>
#include <fmt/color.h>
#include <fmt/format.h>

#include "Configuration.hpp"

ShaderBuild::ShaderBuild()
{
    createSlangSession();

    const auto sources = collectShaders();
    for (const auto& i : sources)
    {
        compileShader(i);
        mSuccess++;
    }
    std::println("Compiled ({}/{}) file(s) ({} stage(s))", mSuccess, sources.size(), mStages);
}

std::vector<std::filesystem::path> ShaderBuild::collectShaders()
{
    std::vector<std::filesystem::path> files;

    for (const auto& file : std::filesystem::directory_iterator(Configuration::getShaderSourceDir()))
    {
        if (file.is_regular_file() && file.path().extension() == ".slang")
        {
            files.push_back(file);
        }
    }

    return files;
}

void ShaderBuild::createSlangSession()
{
    slang::createGlobalSession(mGlobalSession.writeRef());

    const std::array options = {
        slang::CompilerOptionEntry {
            .name  = slang::CompilerOptionName::Capability,
            .value = {
                .kind = slang::CompilerOptionValueKind::Int,
                .intValue0 = mGlobalSession->findCapability("spvDescriptorHeapEXT")
            },
        },
        slang::CompilerOptionEntry {
            .name  = slang::CompilerOptionName::DebugInformation,
            .value = {
                .kind = slang::CompilerOptionValueKind::Int,
                .intValue0 = SLANG_DEBUG_INFO_LEVEL_STANDARD,
            },
        },
        slang::CompilerOptionEntry {
            .name  = slang::CompilerOptionName::DefaultImageFormatUnknown,
            .value = {
                .kind = slang::CompilerOptionValueKind::Int,
                .intValue0 = 1,
            },
        },
    };

    slang::TargetDesc targetDesc = {
        .format                      = SLANG_SPIRV,
        .profile                     = mGlobalSession->findProfile("spirv_1_6"),
        .flags                       = SLANG_TARGET_FLAG_GENERATE_SPIRV_DIRECTLY,
        .forceGLSLScalarBufferLayout = true,
        .compilerOptionEntries       = options.data(),
        .compilerOptionEntryCount    = options.size(),
    };

    std::vector<slang::PreprocessorMacroDesc> macros = {};

    slang::SessionDesc sessionDesc = {
        .targets                 = &targetDesc,
        .targetCount             = 1,
        .defaultMatrixLayoutMode = SLANG_MATRIX_LAYOUT_COLUMN_MAJOR,
        .preprocessorMacros      = macros.data(),
        .preprocessorMacroCount  = static_cast<SlangInt>(macros.size()),
    };

    sessionDesc.targets     = &targetDesc;
    sessionDesc.targetCount = 1;

    mGlobalSession->createSession(sessionDesc, mSession.writeRef());
}

std::expected<std::vector<char>, std::string> ShaderBuild::readShaderFile(const std::filesystem::path& filePath)
{
    std::ifstream file(filePath, std::ios::ate | std::ios::binary);
    if (!file.is_open())
    {
        return std::unexpected { fmt::format("Failed to open file: {}", filePath.string()) };
    }

    const std::streamsize fileSize = file.tellg();
    std::vector<char>     buffer(fileSize);

    file.seekg(0);
    file.read(buffer.data(), fileSize);

    file.close();
    return buffer;
}

void ShaderBuild::compileShader(const std::filesystem::path& source)
{
    const auto src = readShaderFile(source);
    if (!src.has_value())
    {
        std::println("{}", src.error());
        return;
    }

    const std::string sourceStr(src.value().begin(), src.value().end());

    Slang::ComPtr<slang::IModule> module;
    {
        Slang::ComPtr<slang::IBlob> diagnosticsBlob;
        module = mSession->loadModuleFromSourceString(source.filename().string().c_str(), source.string().c_str(), sourceStr.c_str(), diagnosticsBlob.writeRef());

        diag(diagnosticsBlob);
        if (!module)
        {
            std::println("Failed to load shader: {}", source.string());
            return;
        }
    }

    const SlangInt32                               entryPointCount = module->getDefinedEntryPointCount();
    std::vector<Slang::ComPtr<slang::IEntryPoint>> entryPoints(entryPointCount);
    {
        for (SlangInt32 i = 0; i < entryPointCount; ++i)
        {
            if (SLANG_FAILED(module->getDefinedEntryPoint(i, entryPoints[i].writeRef())) || !entryPoints[i])
            {
                std::println("Error getting entry point {}", i);
                return;
            }
        }
    }

    for (const auto& [entryPointIndex, entryPoint] : std::views::enumerate(entryPoints))
    {
        std::array<slang::IComponentType*, 2> componentTypes = { module, entryPoint };

        Slang::ComPtr<slang::IComponentType> composedProgram;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            const SlangResult result = mSession->createCompositeComponentType(componentTypes.data(), componentTypes.size(), composedProgram.writeRef(), diagnosticsBlob.writeRef());
            diag(diagnosticsBlob);
            if (SLANG_FAILED(result))
            {
                std::println("Failed to compile shader: {}", source.string());
                return;
            }
        }

        Slang::ComPtr<slang::IComponentType> linkedProgram;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            const SlangResult result = composedProgram->link(linkedProgram.writeRef(), diagnosticsBlob.writeRef());
            diag(diagnosticsBlob);
            if (SLANG_FAILED(result))
            {
                std::println("Failed to compile shader: {}", source.string());
                return;
            }
        }

        slang::ProgramLayout* layout = linkedProgram->getLayout(0);

        slang::EntryPointReflection* epLayout = layout->getEntryPointByIndex(0);
        SlangStage                   stage    = epLayout->getStage();

        Slang::ComPtr<slang::IBlob> spirvCode;
        {
            Slang::ComPtr<slang::IBlob> diagnosticsBlob;
            const SlangResult result = linkedProgram->getEntryPointCode(0, 0, spirvCode.writeRef(), diagnosticsBlob.writeRef());
            diag(diagnosticsBlob);
            if (SLANG_FAILED(result))
            {
                std::println("Failed to compile shader: {}", source.string());
                return;
            }
        }

        const auto stageExt = stageToExtension(stage);
        if (!stageExt.has_value())
        {
            std::println("Shader stage {} didn't evaluate to any known file extension for entryPoint: {}.", std::to_underlying(stage), epLayout->getName());
            return;
        }

        auto binPath = Configuration::getShaderDir().append(fmt::format("{}.{}.spv", fullStem(source).string(), stageExt.value()));

        writeShader(binPath, spirvCode);

        std::println("{}", fmt::format(
            "[{}] {} -> {}",
            styled("ok", fg(fmt::color::cyan) | fmt::emphasis::bold),
            styled(source.filename().string(), fg(fmt::color::light_gray)),
            binPath.filename().string()
        ));

        mStages++;
    }
}

std::expected<std::string, std::string> ShaderBuild::stageToExtension(const SlangStage stage)
{
    switch (stage)
    {
        case SLANG_STAGE_VERTEX:            return "vert";
        case SLANG_STAGE_HULL:              return "tc";
        case SLANG_STAGE_DOMAIN:            return "te";
        case SLANG_STAGE_GEOMETRY:          return "geom";
        case SLANG_STAGE_FRAGMENT:          return "frag";
        case SLANG_STAGE_COMPUTE:           return "comp";
        case SLANG_STAGE_RAY_GENERATION:    return "rgen";
        case SLANG_STAGE_INTERSECTION:      return "rint";
        case SLANG_STAGE_ANY_HIT:           return "ahit";
        case SLANG_STAGE_CLOSEST_HIT:       return "chit";
        case SLANG_STAGE_MISS:              return "miss";
        case SLANG_STAGE_CALLABLE:          return "call";
        case SLANG_STAGE_MESH:              return "mesh";
        case SLANG_STAGE_AMPLIFICATION:     return "task";
        case SLANG_STAGE_NONE:              // Falls through
        default:                            return std::unexpected { "Invalid shader stage" };
    }
}

void ShaderBuild::writeShader(const std::filesystem::path& binPath, slang::IBlob* spirv)
{
    std::ofstream out(binPath, std::ios::binary);
    out.write(reinterpret_cast<const char*>(spirv->getBufferPointer()), spirv->getBufferSize());
    out.close();
}

void ShaderBuild::diag(slang::IBlob* diagBlob)
{
    if (diagBlob != nullptr)
    {
        std::println("{}", (const char*)diagBlob->getBufferPointer());
    }
}

std::filesystem::path ShaderBuild::fullStem(std::filesystem::path p)
{
    while (p.has_extension())
    {
        p.replace_extension();
    }
    return p.filename();
}
