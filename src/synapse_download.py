import argparse

import synapseclient
import synapseutils


TOKEN_FILE = 'synapse_token'


_ACCEPTED_SYN_IDS = [
    'syn68646516',  # Task 2 - Segmentation
    'syn68737427' # Task 2 - Segmentation Validation
]


def get_token(token_file: str):
    token = None
    with open(token_file) as f:
        token = f.read()
    return token


def main(output_folder: str, synID: str, token_file: str):
    token = get_token(token_file)

    syn = synapseclient.Synapse()
    syn.login(authToken=token)

    synapseutils.syncFromSynapse(syn, synID, path=output_folder)


def entrypoint():
    parser = argparse.ArgumentParser()
    parser.add_argument('output_folder', type=str)
    parser.add_argument('-synid', '--synapse_id', type=str, default='syn68737427')
    parser.add_argument('--token_file', type=str, default=TOKEN_FILE)

    args = parser.parse_args()
    if args.synapse_id not in _ACCEPTED_SYN_IDS:
        raise ValueError(f'Non-supported synapse id code provided. Please provide one of:\n{_ACCEPTED_SYN_IDS}')

    main(args.output_folder, args.synapse_id, args.token_file)



if __name__ == "__main__":
    entrypoint()