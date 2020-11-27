import os
path_anaconda = "C:/Users/brenobeirigo/Anaconda3/Scripts/"
env_name = "env_slevels_new"

tests = [
    {
        "folder_instances": "d:/bb/mod/config/A3_A4_1st_class_distribution/",
        "folder_log": "d:/bb/mod/config/A3_A4_1st_class_distribution/log/",
        "label": "TS_2K-AP-BP-8",
        "config": "TS_2K-AP-BP-8.json"
    },
    {
        "folder_instances": "d:/bb/mod/config/A3_A4_1st_class_distribution/",
        "folder_log": "d:/bb/mod/config/A3_A4_1st_class_distribution/log/",
        "label": "TS_2K-AP-BP-9",
        "config": "TS_2K-AP-BP-9.json"
    },
    {
        "folder_instances": "d:/bb/mod/config/A3_A4_1st_class_distribution/",
        "folder_log": "d:/bb/mod/config/A3_A4_1st_class_distribution/log/",
        "label": "TS_2K-AP-BP-1",
        "config": "TS_2K-AP-BP-1.json"
    },
    {
        "folder_instances": "d:/bb/mod/config/A3_A4_1st_class_distribution/",
        "folder_log": "d:/bb/mod/config/A3_A4_1st_class_distribution/log/",
        "label": "TS_2K-A2-B8-8",
        "config": "TS_2K-A2-B8-8.json"
    },
    {
        "folder_instances": "d:/bb/mod/config/A3_A4_1st_class_distribution/",
        "folder_log": "d:/bb/mod/config/A3_A4_1st_class_distribution/log/",
        "label": "TS_2K-A2-B8-9",
        "config": "TS_2K-A2-B8-9.json"
    },
    {
        "folder_instances": "d:/bb/mod/config/A3_A4_1st_class_distribution/",
        "folder_log": "d:/bb/mod/config/A3_A4_1st_class_distribution/log/",
        "label": "TS_2K-A2-B8-1",
        "config": "TS_2K-A2-B8-1.json"
    },
    # ONLY B CLASS
    {
        "folder_instances": "d:/bb/mod/config/adp_tune/",
        "folder_log": "d:/bb/mod/config/adp_tune/log/",
        "label": "TS_2K-B-1-1",
        "config": "TS_2K-B-1-1.json"
    },
    {
        "folder_instances": "d:/bb/mod/config/adp_tune/",
        "folder_log": "d:/bb/mod/config/adp_tune/log/",
        "label": "TS_2K-B-1-2",
        "config": "TS_2K-B-1-2.json"
    },
    # {
    #     "folder_instances": "d:/bb/mod/config/adp_tune/",
    #     "folder_log": "d:/bb/mod/config/adp_tune/log/",
    #     "label": "TS_2K-B-9-1",
    #     "config": "TS_2K-B-9-1.json"
    # },
    {
        "folder_instances": "d:/bb/mod/config/adp_tune/",
        "folder_log": "d:/bb/mod/config/adp_tune/log/",
        "label": "TS_2K-B-9-2",
        "config": "TS_2K-B-9-2.json"
    },
    {
        "folder_instances": "d:/bb/mod/config/adp_tune/",
        "folder_log": "d:/bb/mod/config/adp_tune/log/",
        "label": "TS_2K-B-1-3",
        "config": "TS_2K-B-1-3.json"
    },
    {
        "folder_instances": "d:/bb/mod/config/adp_tune/",
        "folder_log": "d:/bb/mod/config/adp_tune/log/",
        "label": "TS_2K-B-8-1",
        "config": "TS_2K-B-8-1.json"
    },
    {
        "folder_instances": "d:/bb/mod/config/adp_tune/",
        "folder_log": "d:/bb/mod/config/adp_tune/log/",
        "label": "TS_2K-B-8-2",
        "config": "TS_2K-B-8-2.json"
    },
]

# ## Run single
# python main.py "d:/bb/mod/config/adp_tune/" "TS_2K-B-1-3-chosen"

for instance in tests: 
    folder_log = instance["folder_log"]
    folder_instances = instance["folder_instances"]
    label = instance["label"]
    config = instance["config"]
    path_log = f'{folder_log}{label}.log'

    os.system(
        f'start cmd.exe /k "cd {path_anaconda} '
        f'& activate {env_name} '
        f'& title {label} '
        f'& python main.py {folder_instances} {config} >> {path_log}'
    )