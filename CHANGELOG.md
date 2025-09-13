# [1.10.0](https://github.com/MA-DOS/ShaReComp/compare/v1.9.0...v1.10.0) (2025-09-13)


### Features

* added clustering and prediction endpoints for kcca and random forest regressor ([478b40f](https://github.com/MA-DOS/ShaReComp/commit/478b40f15cc19f99eb42deac8d816acc1c82f2a2))
* added cv + hyperparameter tuning for random forest regressor ([379af05](https://github.com/MA-DOS/ShaReComp/commit/379af05a38b5bd52fda26d9daf5d456237c56f86))

# [1.9.0](https://github.com/MA-DOS/ShaReComp/compare/v1.8.0...v1.9.0) (2025-09-07)


### Bug Fixes

* add correct container permissions for fio benchmark ([b746fa0](https://github.com/MA-DOS/ShaReComp/commit/b746fa0169120f5a36b3836986237b1737022288))
* added correct numa layout for coloc core level experiments ([9d53c71](https://github.com/MA-DOS/ShaReComp/commit/9d53c7168f02ac840824d4fe2250a100d57fa171))


### Features

* added calculation fordistance matrix, clustering threshold and completed clustering function ([ccd4ee2](https://github.com/MA-DOS/ShaReComp/commit/ccd4ee2ed4833f0e742cbe367639c670f41b50d6))
* added logic to compute feature temporal signature for colocated jobs after clustering results ([ba87404](https://github.com/MA-DOS/ShaReComp/commit/ba874041f10a994878811bfeb958466a723731ab))
* added pattern feature as only temporal signature for a job as comparison with model performance and clustering results ([95cee63](https://github.com/MA-DOS/ShaReComp/commit/95cee6366e4a99b85be8b14f2496feeae180c399))

# [1.8.0](https://github.com/MA-DOS/ShaReComp/compare/v1.7.0...v1.8.0) (2025-07-07)


### Features

* added custom distance function, sketched clustering algorithm, implemented Random Forest training and predictor ([7b59807](https://github.com/MA-DOS/ShaReComp/commit/7b59807856c9a8f8b1fb9166581ce23f9df21c9b))
* refined memory and fileio benchmarks to run full workload, integrate mean power values, recompute affinity scores and output result ([0dc1b8f](https://github.com/MA-DOS/ShaReComp/commit/0dc1b8f08c0da19fc2892d9d8b640414a0d887ad))

# [1.7.0](https://github.com/MA-DOS/ShaReComp/compare/v1.6.0...v1.7.0) (2025-06-30)


### Features

* added kcca prediction function on test data ([2f8ca13](https://github.com/MA-DOS/ShaReComp/commit/2f8ca138489e82e0ed0b60773ad0ade4378e54ea))

# [1.6.0](https://github.com/MA-DOS/ShaReComp/compare/v1.5.0...v1.6.0) (2025-06-30)


### Features

* refined scripts into funcs, added scope-specific data processing, added power values to feature matrix, added annotations for interpretability, ([82dfc70](https://github.com/MA-DOS/ShaReComp/commit/82dfc70758761294bc8bbc31507e96ce0c81549d))

# [1.5.0](https://github.com/MA-DOS/ShaReComp/compare/v1.4.1...v1.5.0) (2025-06-24)


### Features

* added power measurements into benchmarking pipeline ([9bd946c](https://github.com/MA-DOS/ShaReComp/commit/9bd946cd53c69b244f8a31f41f466a5b9a69f865))

## [1.4.1](https://github.com/MA-DOS/ShaReComp/compare/v1.4.0...v1.4.1) (2025-06-09)


### Bug Fixes

* fix affinity score dict typo ([21c41a5](https://github.com/MA-DOS/ShaReComp/commit/21c41a5e3e638c704b2429baadc99e2b1ff0b221))

# [1.4.0](https://github.com/MA-DOS/ShaReComp/compare/v1.3.0...v1.4.0) (2025-06-09)


### Features

* added feature preprocessing and fitted kcca model + increased benchmark intensity for affinity score comp ([ad183c2](https://github.com/MA-DOS/ShaReComp/commit/ad183c2981aeb4fd7ffaa7ad7d88dd035171bf73))

# [1.3.0](https://github.com/MA-DOS/ShaReComp/compare/v1.2.0...v1.3.0) (2025-06-06)


### Features

* add linpack to cpu bench and feature vector computation per workflow task ([a4e8ae3](https://github.com/MA-DOS/ShaReComp/commit/a4e8ae3e36e0b824416a685a240fb2547aded14f))

# [1.2.0](https://github.com/MA-DOS/ShaReComp/compare/v1.1.0...v1.2.0) (2025-05-30)


### Features

* added affinity score computation and did entity matching for nxf containers to nf-core tasks in time-series data ([e87e647](https://github.com/MA-DOS/ShaReComp/commit/e87e64707236318569f7ce92fc0b64bfedd1bbca))

# [1.1.0](https://github.com/MA-DOS/ShaReComp/compare/v1.0.0...v1.1.0) (2025-05-30)


### Features

* add wip affn score computation and monitoring time-series entity matching ([c7afe0d](https://github.com/MA-DOS/ShaReComp/commit/c7afe0d7299b80dd710d73479b0a43b2a8176508))

# 1.0.0 (2025-05-30)


### Features

* add versioning ci ([81391fb](https://github.com/MA-DOS/ShaReComp/commit/81391fb23f0ed183a128d3a1df2e02556189775b))
* added colocated workloads ([d9c6c28](https://github.com/MA-DOS/ShaReComp/commit/d9c6c28fcb89d7653a04f422986b29674ddff06d))
* Initial skeleton ([561cdde](https://github.com/MA-DOS/ShaReComp/commit/561cddec4e5a08f6cc4a41fb932b65b514ab0207))
