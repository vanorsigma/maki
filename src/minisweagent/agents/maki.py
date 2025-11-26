from minisweagent.agents.default import DefaultAgent
from typing import Callable

class MakiAgent(DefaultAgent):
    def __init__(self, *args, on_observation_callback: Callable[[str], None], on_output_callback: Callable[[str], None] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._on_observation_callback = on_observation_callback
        self._on_output_callback = on_output_callback

    def get_observation(self, response: dict) -> dict:
        parsed_action = self.parse_action(response)
        if self._on_observation_callback:
            self._on_observation_callback(parsed_action['action'])

        output = self.execute_action(parsed_action)
        observation = self.render_template(self.config.action_observation_template, output=output)
        self.add_message("user", observation)
        if self._on_output_callback:
            self._on_output_callback(output['output'])
        return output
