from glob import glob
from setuptools import find_packages, setup

package_name = "fr3_husky_nn_policy"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            glob("config/*.yaml") + glob("config/*.json"),
        ),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/models", glob("models/*.npz") + glob("models/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "matplotlib"],
    zip_safe=True,
    maintainer="DYROS",
    maintainer_email="dyros@snu.ac.kr",
    description="CPU NumPy policy deployment for FR3 Husky robots.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ppo_liftcube_policy_node = fr3_husky_nn_policy.ppo_liftcube_policy_node:main",
            "ppo_reach_policy_node = fr3_husky_nn_policy.ppo_reach_policy_node:main",
            "fr3_sysid_probe_node = fr3_husky_nn_policy.fr3_sysid_probe_node:main",
            "reach_target_cli = fr3_husky_nn_policy.reach_target_cli:main",
            "export_ppo_actor = fr3_husky_nn_policy.export_ppo_actor:main",
        ],
    },
)
