import argparse

import synapseclient
import synapseutils


_ACCEPTED_SYN_IDS = {
    'training': 'syn68646516',  # Task 2 - Training
    'validation': 'syn68737427' # Task 2 - Validation
}


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

    parser.add_argument('-synid', '--synapse_id', type=str,
                        choices=_ACCEPTED_SYN_IDS.keys(),
                        default='training')

    parser.add_argument('--token_file', type=str, default='synapse_token')

    args = parser.parse_args()

    main(args.output_folder, _ACCEPTED_SYN_IDS[args.synapse_id], args.token_file)


if __name__ == "__main__":
    entrypoint()