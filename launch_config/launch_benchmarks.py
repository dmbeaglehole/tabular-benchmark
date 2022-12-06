from utils import create_sweep
import pandas as pd

# We use one project per benchmark to avoid WandB getting super slow
WANDB_PROJECT_NAMES = ["feature_kernels_2"]


data_transform_config = {
    "data__method_name": {
        "value": "real_data"
    },
    "n_iter": {
        "value": "auto",
    },
}

benchmarks = [
#            {"task": "regression",
#                    "dataset_size": "medium",
#                    "categorical": False,
#                    "datasets":  ["cpu_act",
#                      "pol"]},
#
#           {"task": "regression",
#                   "dataset_size": "medium",
#                   "categorical": False,
#                   "datasets":  ["elevators",
#                     "isolet"]
#            },
#
#           {"task": "regression",
#                   "dataset_size": "medium",
#                   "categorical": False,
#                   "datasets":  ["wine_quality",
#                      "Ailerons",
#                      "houses",
#                      "house_16H"]},
#
#              {"task": "regression",
#                   "dataset_size": "medium",
#                   "categorical": False,
#                   "datasets":  ["diamonds",
#                      "Brazilian_houses"]},
#
#              {"task": "regression",
#                   "dataset_size": "medium",
#                   "categorical": False,
#                   "datasets":  ["Bike_Sharing_Demand",
#                      "nyc-taxi-green-dec-2016"]},
#
#              {"task": "regression",
#                   "dataset_size": "medium",
#                   "categorical": False,
#                   "datasets":  ["house_sales",
#                      "sulfur"]},
#
#              {"task": "regression",
#                   "dataset_size": "medium",
#                   "categorical": False,
#                   "datasets":  ["medical_charges",
#                      "MiamiHousing2016",
#                      "superconduct"]},
#
#              {"task": "regression",
#                   "dataset_size": "medium",
#                   "categorical": False,
#                   "datasets":  ["california",
#                      "fifa"]},
#    
#            {"task": "regression",
#                   "dataset_size": "medium",
#                   "categorical": False,
#                   "datasets":  ["year"]},

                 {"task": "regression",
                    "dataset_size": "large",
                    "categorical": False,
                    "datasets": ["diamonds"]
                 },

                 {"task": "regression",
                    "dataset_size": "large",
                    "categorical": False,
                    "datasets": ["nyc-taxi-green-dec-2016"]
                 },
    
                {"task": "regression",
                    "dataset_size": "large",
                    "categorical": False,
                    "datasets": ["year"]
                 },

#                {"task": "classif",
#                    "dataset_size": "medium",
#                    "categorical": False,
#                    "datasets": ["electricity",
#                                 "covertype",
#                                 "pol"]},
#
#                {"task": "classif",
#                    "dataset_size": "medium",
#                    "categorical": False,
#                    "datasets": ["house_16H",
#                                 "kdd_ipums_la_97-small"]},
#              
#               {"task": "classif",
#                    "dataset_size": "medium",
#                    "categorical": False,
#                    "datasets": ["MagicTelescope",
#                                 "bank-marketing",
#                                 "phoneme"]},
#
#               {"task": "classif",
#                    "dataset_size": "medium",
#                    "categorical": False,
#                    "datasets": ["MiniBooNE",
#                                 "Higgs"]},
#              
#               {"task": "classif",
#                    "dataset_size": "medium",
#                    "categorical": False,
#                    "datasets": ["eye_movements",
#                                 "jannis"]},
#
#               {"task": "classif",
#                    "dataset_size": "medium",
#                    "categorical": False,
#                    "datasets": ["credit",
#                                 "california",
#                                 "wine"]},
                

                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": False,
                    "datasets": ["MiniBooNE"]
                },

                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": False,
                    "datasets": ["covertype"]
                },

                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": False,
                    "datasets": ["Higgs",
                                 "jannis"]
                 },

                #{"task": "regression",
                #    "dataset_size": "medium",
                #    "categorical": True,
                # "datasets": ["yprop_4_1"]},
    
                #{"task": "regression",
                #    "dataset_size": "medium",
                #    "categorical": True,
                # "datasets": ["analcatdata_supreme"]},

                #{"task": "regression",
                #    "dataset_size": "medium",
                #    "categorical": True,
                # "datasets": ["visualizing_soil",
                #             "black_friday",
                #             "nyc-taxi-green-dec-2016"]},
              
                # {"task": "regression",
                #    "dataset_size": "medium",
                #    "categorical": True,
                # "datasets": ["diamonds",
                #             "Mercedes_Benz_Greener_Manufacturing"]},

                # {"task": "regression",
                #    "dataset_size": "medium",
                #    "categorical": True,
                # "datasets": ["Brazilian_houses",
                #            "Bike_Sharing_Demand"]},
              
                # {"task": "regression",
                #    "dataset_size": "medium",
                #    "categorical": True,
                # "datasets": ["OnlineNewsPopularity",
                #             "house_sales"]},

                # {"task": "regression",
                #    "dataset_size": "medium",
                #    "categorical": True,
                # "datasets": ["particulate-matter-ukair-2017",
                #             "SGEMM_GPU_kernel_performance"]},

                {"task": "regression",
                 "dataset_size": "large",
                 "categorical": True,
                 "datasets": ["black_friday"]
                },

                {"task": "regression",
                 "dataset_size": "large",
                 "categorical": True,
                 "datasets": ["nyc-taxi-green-dec-2016"]
                },
                {"task": "regression",
                 "dataset_size": "large",
                 "categorical": True,
                 "datasets": ["diamonds"]
                },
                {"task": "regression",
                 "dataset_size": "large",
                 "categorical": True,
                 "datasets": ["particulate-matter-ukair-2017"]
                 },
                {"task": "regression",
                 "dataset_size": "large",
                 "categorical": True,
                 "datasets": ["SGEMM_GPU_kernel_performance"]
                 },
                #{"task": "classif",
                #    "dataset_size": "medium",
                #    "categorical": True,
                #    "datasets": ["electricity",
                #                "eye_movements",
                #                  "KDDCup09_upselling"]},

                #{"task": "classif",
                #    "dataset_size": "medium",
                #    "categorical": True,
                #    "datasets": ["covertype",
                #                  "rl"]},

                #{"task": "classif",
                #    "dataset_size": "medium",
                #    "categorical": True,
                #    "datasets": ["road-safety",
                #                  "compass"]},

                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": True,
                    "datasets": ["covertype"]
                },

                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": True,
                    "datasets": ["road-safety"]
                }
]

#models = ["gbt", "rf", "xgb", "hgbt",
#          "ft_transformer", "resnet", "mlp", "saint"]
models = ["ours"]

if __name__ == "__main__":
    sweep_ids = []
    names = []
    projects = []
    for i, benchmark in enumerate(benchmarks):
        for model_name in models:
            for default in [True]:
                name = f"{model_name}_{benchmark['task']}_{benchmark['dataset_size']}"
                if benchmark['categorical']:
                    name += "_categorical"
                else:
                    name += "_numerical"
                if default:
                    name += "_default"
                sweep_id = create_sweep(data_transform_config,
                             model_name=model_name,
                             regression=benchmark["task"] == "regression",
                             categorical=benchmark["categorical"],
                             dataset_size = benchmark["dataset_size"],
                             datasets = benchmark["datasets"],
                             default=default,
                             project=WANDB_PROJECT_NAMES[0],
                             name=name)
                sweep_ids.append(sweep_id)
                names.append(name)
                projects.append(WANDB_PROJECT_NAMES[0])
                print(f"Created sweep {name}")
                print(f"Sweep id: {sweep_id}")
                print(f"In project {WANDB_PROJECT_NAMES[0]}")

    df = pd.DataFrame({"sweep_id": sweep_ids, "name": names, "project":projects})
    df.to_csv("launch_config/sweeps/benchmark_sweeps.csv", index=False)
    print("Check the sweeps id saved at sweeps/benchmark_sweeps.csv")


