#include <iostream>

#include "Core/Application.hpp"
#include "Core/ShaderBuild.hpp"

int main()
{
    std::ignore = ShaderBuild();

    auto app = std::make_unique<Application>();
    gApplication = app.get();

    app->run();

    app.reset();

    return 0;
}
