import json

class Util:

    @staticmethod
    def load_intents(path: str):
        try:
            print(f'loading intents from: {path}')
            with open(path) as file:
                return json.load(file)

        except Exception as e:
            print(f'Error loading intents: {e}')
            raise


    @staticmethod
    def handleSignal(receivedSignal, frame):
        print(f'received signal: {receivedSignal}... exiting app.')
        exit(0)