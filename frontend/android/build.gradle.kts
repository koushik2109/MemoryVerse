allprojects {
    repositories {
        google()
        mavenCentral()
    }
    configurations.all {
        resolutionStrategy {
            force("androidx.core:core-ktx:1.13.1")
            force("androidx.core:core:1.13.1")
        }
    }
}

val newBuildDir: Directory =
    rootProject.layout.buildDirectory
        .dir("../../build")
        .get()
rootProject.layout.buildDirectory.value(newBuildDir)

subprojects {
    val newSubprojectBuildDir: Directory = newBuildDir.dir(project.name)
    project.layout.buildDirectory.value(newSubprojectBuildDir)
    
    afterEvaluate {
        if (project.plugins.hasPlugin("com.android.library") || project.plugins.hasPlugin("com.android.application")) {
            val android = project.extensions.getByName("android") as com.android.build.gradle.BaseExtension
            android.compileSdkVersion(36)
        }
    }
}
subprojects {
    project.evaluationDependsOn(":app")
}

tasks.register<Delete>("clean") {
    delete(rootProject.layout.buildDirectory)
}
