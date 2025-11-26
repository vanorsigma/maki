#!/usr/bin/env python3

import asyncio
import websockets.sync.client
import websockets.sync.connection
import pydantic
import whisper
import numpy as np
import speech_recognition as sr
import time
import torch

import os
import traceback
from pathlib import Path
from typing import Any, Literal, Callable, cast

import typer
import yaml
from datetime import datetime, timedelta
from queue import Queue
from prompt_toolkit.history import FileHistory
from prompt_toolkit.shortcuts import PromptSession
from rich.console import Console
from multiprocessing import Process
from multiprocessing.managers import BaseManager

from minisweagent import global_config_dir
from minisweagent.agents.maki import MakiAgent
from minisweagent.config import builtin_config_dir, get_config_path
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model
from minisweagent.run.extra.config import configure_if_first_time
from minisweagent.utils.log import logger

class MakiOutputWSRequest(pydantic.BaseModel):
    """
    Maki LLM output
    """
    type: Literal["makioutputrequest"] = "makioutputrequest"
    output_type: Literal["output"] | Literal["command"]
    output: str

class MakiOutputThinkingRequest(pydantic.BaseModel):
    """
    Maki LLM output
    """
    type: Literal["makioutputthinkingrequest"] = "makioutputthinkingrequest"
    thinking: bool

class AudioQueueManager(BaseManager):
    def get_audio_queue(self) -> Queue[bytes]:
        """
        This should be overridden later by register()
        """
        raise NotImplementedError()

    def get_result_queue(self) -> Queue[str]:
        """
        This should be overriden later by register()
        """
        raise NotImplementedError()

AudioQueueManager.register('get_audio_queue')
AudioQueueManager.register('get_result_queue')


# Agent config
DEFAULT_CONFIG = Path(
    os.getenv("MSWEA_MINI_CONFIG_PATH", builtin_config_dir / "maki.yaml")
)
DEFAULT_OUTPUT = global_config_dir / "last_mini_run.traj.json"
console = Console(highlight=False)
app = typer.Typer(rich_markup_mode="rich")
prompt_session = PromptSession(
    history=FileHistory(global_config_dir / "mini_task_history.txt")
)
model_name = "codeqwen:v1.5"

# Thingy
WS_SENDER_URL = "ws://127.0.0.1:3001/senders"
WS_RECEIVERS_URL = "ws://127.0.0.1:3001/receivers"
voice_mutex = asyncio.Lock()
voice_locked = False  # locked after at least one checkin is cleared
voice_condition = asyncio.Condition(voice_mutex)

def initialize_model_handler(sender: Callable[[str], None]) -> Callable[[str], None]:
    # fmt: on
    configure_if_first_time()
    config_path = get_config_path(DEFAULT_CONFIG)
    console.print(f"Loading agent config from [bold green]'{config_path}'[/bold green]")
    config = yaml.safe_load(config_path.read_text())

    def __observation_callback(message: str) -> None:
        output = MakiOutputWSRequest(output_type="command", output=message)
        console.print('Sending', output)
        sender(output.model_dump_json())

    def __output_callback(message: str) -> None:
        output = MakiOutputWSRequest(output_type="output", output=message)
        console.print('Sending', output)
        sender(output.model_dump_json())

    # config.setdefault("agent", {})["mode"] = "yolo"
    model = get_model(model_name, config.get("model", {}))
    env = DockerEnvironment(image="python:3.11-slim")
    agent = MakiAgent(
        model, env, on_observation_callback=__observation_callback, on_output_callback=__output_callback, **config.get("agent", {}))

    def __callable(prompt: str) -> None:
        console.print('Prompt is: ', prompt)
        exit_status, result, extra_info = None, None, None
        try:
            exit_status, result = agent.run(
                prompt,
                system="Linux",
                release="6.1.0-37-amd64",
                version="#1 SMP PREEMPT_DYNAMIC Debian 6.1.140-1 (2025-05-22)",
                machine="x86_64")  # type: ignore[arg-type]
        except Exception as e:
            logger.error(f"Error running agent: {e}", exc_info=True)
            exit_status, result = type(e).__name__, str(e)
            extra_info = {"traceback": traceback.format_exc()}

    return __callable

manager = AudioQueueManager(address=('localhost', 50000), authkey=b'abuh')

def server_handler():
    """
    Starts BaseManager. Only manages the queues
    """
    audio_queue = Queue()
    result_queue = Queue()

    AudioQueueManager.register('get_audio_queue', callable=lambda: audio_queue)
    AudioQueueManager.register('get_result_queue', callable=lambda: result_queue)
    s = manager.get_server()
    s.serve_forever()

def audio_receiver_handler():
    """
    Starts BaseManager, receives audio and does model things
    """
    manager.connect()
    model = whisper.load_model("small")

    while True:
        console.log(f"Audio queue is empty? {manager.get_audio_queue().empty()}")
        if manager.get_audio_queue().empty():
            time.sleep(0.25)
            continue

        phrase_bytes = manager.get_audio_queue().get()
        audio_np = (
            np.frombuffer(phrase_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        )

        result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
        text = cast(str, result["text"]).strip()
        console.print('Heard', text)
        manager.get_result_queue().put(text)

def audio_sender_handler(sender: websockets.sync.connection.Connection):
    """
    Sends audio via BaseManager, hopefully to somewhere people are aware of
    """
    manager.connect()

    phrase_time = None
    phrase_bytes = bytes()
    record_timeout = 10
    phrase_timeout = 5
    recognizer = sr.Recognizer()
    remote_queue = manager.get_audio_queue()
    data_queue = Queue()

    model_handler = initialize_model_handler(
        lambda x: manager.get_result_queue().put(x))

    source = sr.Microphone(sample_rate=16000)
    if not source:
        console.print("no mic found")
        return

    with source:
        recognizer.adjust_for_ambient_noise(source)

    def record_callback(_, audio: sr.AudioData) -> None:
        data = audio.get_raw_data()
        data_queue.put(data)

    recognizer.listen_in_background(
        source, record_callback, phrase_time_limit=record_timeout
    )

    while True:
        now = datetime.utcnow()

        # result queue is whatever heard by the whisper model, process it
        result_queue = manager.get_result_queue()
        if not result_queue.empty():
            output = MakiOutputThinkingRequest(thinking=True)
            sender.send(output.model_dump_json())
            model_handler(result_queue.get())
            output = MakiOutputThinkingRequest(thinking=False)
            sender.send(output.model_dump_json())

        # audio related
        if not data_queue.empty():
            phrase_complete = False
            if phrase_time and now - phrase_time > timedelta(seconds=phrase_timeout):
                phrase_complete = True
            phrase_time = now

            audio_data = b"".join(data_queue.queue)
            data_queue.queue.clear()

            phrase_bytes += audio_data

            if phrase_complete:
                console.print("Sending result")
                remote_queue.put(phrase_bytes)
                phrase_bytes = bytes()
        else:
            time.sleep(0.25)


# fmt: off
@app.command()
def main(mic_mode: bool = typer.Option(False, "-m", "--mic", help="Enable mic mode"),
         model_mode: bool = typer.Option(False, "-l", "--model", help="Enable model mode"),
         server_mode: bool = typer.Option(False, "-s", "--server", help="Enable server mode")) -> Any:

    if not mic_mode and not model_name and not server_mode:
        console.print("Specify a mode.")
        return

    if mic_mode:
        sender = websockets.sync.client.connect(WS_SENDER_URL)
        audio_sender_handler(sender)
        sender.close()
        return

    if model_mode:
        audio_receiver_handler()
        return

    if server_mode:
        server_handler()
        return

if __name__ == "__main__":
    app()
