# Teacher annotators model gauge plugin
**This plugin is in alpha mode. Expect there to be issues. Please reach out to the engineering team to resolve**
This command line tool depends on modelgauge and can help with the following
1. Generate pseudo-ground-truth labels using teacher models
2. Measure safety model effectiveness using golden eval sets

## Running teacher model labeling
1. TODO initializing modelgauge as a submodule after cloning
1. cd ../teachers
1. Run `poetry install` from inside the `/teachers` directory
1. Modify `teachers/config/secrets.toml`. Add together api secret (if using together api annotators)
    1. open config/secrets.toml, under [together] api-key add your together api key
1. Run `poetry run modelgauge run-annotators -a <NAME_OF_ANNOTATOR> <CSV_FILE>`
    1. `<NAME_OF_ANNOTATOR>`: Current annotators supported: `llama_3_70b`, `mistral_8x22b`, `llama_guard_2`
    1. `<CSV_FILE>`: csv file must adhere to a certain format (TODO provide reference).
        1. Requires csv file has the following columns: `UID`, `Prompt`, `Response`, `SUT`
        1. Currently (as of July 20, 2024) the requirements are indicated in the `annotation_pipeline.py::CsvAnnotatorInput` class definition. 
    