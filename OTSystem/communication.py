import numpy.typing as npt
import numpy as np

class Packet:
    def __init__(self, message :npt.NDArray[np.float32], malicious: bool = False, tag: int | None = None):
        self.malicious = malicious
        self.detected = False
        self.message = message
        self.tag = tag

class CommunicationChannel:
    """
    the communication environment is treated as a model that is updated in time
    content of the channel is represented as a stack of packets with most recent on top
    send packet: adds packet to the communication stack (top)
    traffic: when a packet is sent the traffic outputs that packet
    receive packet: removes the bottom packet from the stack and outputs it
    receive all: removes all packets from the stack and outputs them
    """

    def __init__(self, communication_frequency: float, communication_noise_std: float = 0.0):
        self.stack = [] # queue of packets, with the most recent packet on top
        self.traffic = None
        self.time_elapsed = 0
        self.communication_frequency = communication_frequency
        self.communication_noise_std = communication_noise_std

    @property
    def info(self):
        return {
            "communication_frequency": self.communication_frequency,
            "communication_noise_std": self.communication_noise_std
        }

    def __str__(self):
        return str(self.info)

    def reset(self):
        self.stack.clear()
        self.traffic = None
        self.time_elapsed = 0

    def send_packet(self, packet : Packet | None, add_to_traffic = True):
        if packet is not None:
            self.stack.append(packet)
            if add_to_traffic:
                self.traffic = packet

    def replace_packet(self, packet : Packet):
        self.stack.clear()
        self.send_packet(packet, add_to_traffic=False)

    def listen2traffic(self)-> Packet | None:
        return self.traffic

    def receive_packet(self)-> Packet:
        if len(self.stack) > 0:
            return self.stack.pop(0)
        else:
            return Packet(message=np.array([0,0], dtype=np.float32), malicious=False)

    def receive_all(self)-> list[Packet]:
        packets = self.stack.copy()
        self.stack.clear()
        return packets
    
    def progress_time(self, dt):
        self.time_elapsed += dt



    