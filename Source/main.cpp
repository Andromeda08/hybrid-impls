#include <iostream>

#include "Core/Application.hpp"
#include "Core/ShaderBuild.hpp"

int main(const int32_t argc, const char** argv)
{
    std::ignore = ShaderBuild();

    // Check if shader compilation only run
    for (int i = 0; i < argc; i++)
    {
        if (std::string_view{argv[i]} == "-s")
        {
            return 0;
        }
    }

    auto app = std::make_unique<Application>();
    gApplication = app.get();

    app->run();

    app.reset();

    return 0;
}
