cc_binary(
	name = "2d-mle-finite-difference-data",
	srcs = ["2d-mle-finite-difference.cpp"],
	deps = ["//src/brownian-motion:2d-brownian-motion",
	        "//src/mle-estimator:2d-mle-finite-difference",
		"//src/mle-estimator:2d-mle-method-of-images"],
	copts = ["-Isrc/images-expansion/",
		 "-Isrc/finite-difference-arpack-version-2/"],
)

cc_binary(
	name = "2d-mle-finite-difference-data-1-14",
	srcs = ["2d-mle-finite-difference-1-14.cpp"],
	deps = ["//src/brownian-motion:2d-brownian-motion",
	        "//src/mle-estimator:2d-mle-finite-difference",
		"//src/mle-estimator:2d-mle-method-of-images"],
	copts = ["-Isrc/images-expansion/",
		 "-Isrc/finite-difference-arpack-version-2/"],
)

cc_binary(
	name = "2d-mle-finite-difference-data-15-28",
	srcs = ["2d-mle-finite-difference-15-28.cpp"],
	deps = ["//src/brownian-motion:2d-brownian-motion",
	        "//src/mle-estimator:2d-mle-finite-difference",
		"//src/mle-estimator:2d-mle-method-of-images"],
	copts = ["-Isrc/images-expansion/",
		 "-Isrc/finite-difference-arpack-version-2/"],
)

cc_binary(
	name = "2d-mle-finite-difference-data-29-36",
	srcs = ["2d-mle-finite-difference-29-36.cpp"],
	deps = ["//src/brownian-motion:2d-brownian-motion",
	        "//src/mle-estimator:2d-mle-finite-difference",
		"//src/mle-estimator:2d-mle-method-of-images"],
	copts = ["-Isrc/images-expansion/",
		 "-Isrc/finite-difference-arpack-version-2/"],
)

cc_binary(
	name = "2d-mle-finite-difference-data-37-48",
	srcs = ["2d-mle-finite-difference-37-48.cpp"],
	deps = ["//src/brownian-motion:2d-brownian-motion",
	        "//src/mle-estimator:2d-mle-finite-difference",
		"//src/mle-estimator:2d-mle-method-of-images"],
	copts = ["-Isrc/images-expansion/",
		 "-Isrc/finite-difference-arpack-version-2/"],
)

cc_binary(
	name = "2d-mle-finite-difference-test",
	srcs = ["2d-mle-finite-difference-test.cpp"],
	deps = ["//src/brownian-motion:2d-brownian-motion",
	        "//src/mle-estimator:2d-mle-finite-difference"],
	copts = ["-Isrc/nlopt/api",
		 "-Isrc/finite-difference-arpack-igraph",
		 "-O"],
)

cc_binary(
	name = "2d-mle-method-of-images-test",
	srcs = ["2d-mle-method-of-images-test.cpp"],
	deps = ["//src/brownian-motion:2d-brownian-motion",
	        "//src/mle-estimator:2d-mle-finite-difference",
		"//src/mle-estimator:2d-mle-method-of-images"],
)

cc_library(
	name = "2d-mle-finite-difference",
	srcs = ["2DMLEFiniteDifference.cpp"],
	hdrs = ["2DMLEFiniteDifference.hpp"],
	visibility = ["//visibility:public"],
	deps = ["//src/finite-difference-arpack-igraph:2d-heat-equation-finite-difference",
		"//src/igraph-0.7.1:igraph",
		"//src/nlopt:nlopt"],
	linkopts = ["-lm", "-fopenmp"],
	copts = ["-O",
		 "-fopenmp",
		 "-Isrc/nlopt/api",
	         "-Isrc/images-expansion",
		 "-Isrc/finite-difference-arpack-igraph",
		 "-Isrc/igraph-0.7.1/include"],
)

cc_library(
	name = "2d-mle-method-of-images",
	srcs = ["2DMLEMethodOfImages.cpp"],
	hdrs = ["2DMLEMethodOfImages.hpp"],
	deps = ["//src/finite-difference-arpack-igraph:pde-data-types",
	        "//src/images-expansion:1d-advection-diffusion-images",
		"//src/nlopt:nlopt"],
	copts = ["-Isrc/nlopt/api", "-Isrc/images-expansion/"],
	visibility = ["//visibility:public"],
)
