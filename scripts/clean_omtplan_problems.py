import os

OMTPLAN_DIRECTORY = "data/benchmark/omtplan/h2"

STR_TO_REMOVE = "--------------------------------------------------------------------------------"


def clean() -> None:
    for root, _, files in os.walk(OMTPLAN_DIRECTORY):
        for file in files:
            if file.endswith(".smt2"):
                full_path = os.path.join(root, file)
                with open(full_path, 'r') as f:
                    content = f.read()
                content = content.replace(STR_TO_REMOVE, "")
                with open(full_path, 'w') as f:
                    f.write(content)


if __name__ == "__main__":
    clean()
