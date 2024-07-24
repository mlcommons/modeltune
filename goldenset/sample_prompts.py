import argparse
import pandas as pd
import os
import subprocess

SEED = 42

INPUT_COLS = ['UID', 'Locale', 'Text', 'Hazard', 'Hazard Subtype']

SUTS = [
    'gemma-7b-it',
    'Mixtral-8x7B-Instruct-v0.1',
    'llama-3-70b-chat-hf',
    'WizardLM-13B-V1.2',
    'Qwen1.5-72B-Chat',
    'alpaca-7b',
    'vicuna-13b-v1.5',
    'openchat-3.5-1210',
    'llama-2-70b-chat',
    'llama-2-13b-chat-hf',
    'Mistral-7B-Instruct-v0.2',
    'StripedHyena-Nous-7B',
    'llama-2-7b-chat-hf',
    'chronos-hermes-13b',
    'deepseek-llm-67b-chat']


HAZARDS_v1 =[
     'Non-violent crime',
     'Specialized Advice',
     'Intellectual Property',
     'Privacy',
     'Defamation',
     'Sexual Content',
     'Suicide & Self-harm',
     'Child Sexual Exploitation',
     'Hate',
     'Indiscriminate Weapons (CBRNE)',
     'Sex crimes',
     'Violent crimes'
]

def cmd(prompt_set):
    """
    Run modelgauge to get results for SUTS on prompt-set
    """
    suts = ['--sut {sut}' for sut in SUTS]
    subprocess.Popen(['modelgauge', 'run-prompts', '--workers', '40' '--cache', 'mycache', suts, prompt_set])

def merge_prompt_respose(prompt_csv: str, response_csv: str):
    """ 
    Merge prompt_csv and response_csv on UID to get all data from prompt_csv
    """
    prompt_df = pd.read_csv(prompt_csv)
    response_df = pd.read_csv(response_csv)
    output_df = response_df.merge(prompt_df[['UID', 'Hazard', 'Locale', 'Hazard Subtype']], on="UID")
    output_df = output_df[INPUT_COLS + SUTS]
    output_df.drop_duplicates(subset=['Text'], keep='first', inplace=True)
    return output_df

def filter_hazard_v1_and_locale(df: pd.DataFrame, locale=None):
    """
    """
    idx = df['Hazard'].apply(lambda x: x in HAZARDS_v1)
    df = df[idx]
    if locale is not None:
        df = df[df['Locale'] == 'en-US']
    return df

def expand_suts_as_rows(df: pd.DataFrame):
    """
    Expand response values form suts to rows with "
    """
    
    outputs = []
    for sut in SUTS:
        output = df[INPUT_COLS + [sut]].copy()
        output['sut'] = sut
        output['response'] =  df[sut]
        outputs.append(output)
    return pd.concat(outputs)[INPUT_COLS + ['sut', 'response']]

def sample_prompts_by_uid(df, equal_hazards=False):
    """
    Sample prompts by UID such that there is sut per response
    """
    samples = (df
           .groupby(['UID'])
           .apply(lambda x: x.sample(1), include_groups=False)
           .reset_index(level=0)
          )
    if equal_hazards:
        min_hazard_count = df.groupby(['sut', 'Hazard'])['Hazard'].count().min()
        samples = (samples
                   .groupby(['Hazard'])
                   .apply(lambda x: x.sample(min_hazard_count), include_groups=False)
                   .reset_index(level=0)
                  )
    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Clean and sample prompts from modelgauge"
    )

    parser.add_argument(
            "--prompts", type=str, help="Path to the prompt csv", required=True
        )
    parser.add_argument(
        "--responses", type=str, help="Path to the response csv", required=True
    )

    parser.add_argument(
        "--output", type=str, help="Path to the output csv", required=False
    )
    
    parser.add_argument(
        "--equal_hazards", type=bool,
        help="Sample prompts to have have equal distribution for each hazard", nargs='?', const=False, required=False
    )

    args = parser.parse_args()

    output = merge_prompt_respose(args.prompts, args.responses)
    output = filter_hazard_v1_and_locale(output, locale='en-US') 
    output_all = expand_suts_as_rows(output)
    output_sampled = sample_prompts_by_uid(output_all, equal_hazards=args.equal_hazards)
    output_sampled.to_csv(args.output, index=False)
    
if __name__ == '__main__':
    main()
