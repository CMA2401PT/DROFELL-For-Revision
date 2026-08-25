import socket
import extend_asyncio as asyncio
import struct
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Type, TypeVar, Union, cast
from uuid import uuid4
import numpy as np

from extend_asyncio.common import FrameConn, Frames
from .faster_serialize.pickle import FasterPickler, FasterUnpickler
from ..core import Context, OrbitName, Packet, LocationByName
from .define import Extender, LinkPerfDescribe, PacketTypeNodeDrop, RecvPref, SendPerf
from typing import List
import psutil


def get_all_ips() -> List[str]:
    ips: List[str] = []
    for if_addrs in psutil.net_if_addrs().values():
        for addr in if_addrs:
            if addr.family == socket.AF_INET:
                ips.append(addr.address)
    return ips


_Ret = TypeVar('_Ret')


def serialize(typ: str, time_stamp: int, data: Tuple[bool, int], datas: List[Tuple[bool, int]]) -> bytes:
    # data: if not list
    # datas: if list
    typ_bytes = typ.encode('utf-8')
    typ_len = len(typ_bytes)
    datas = [data]+datas
    header = struct.pack(f'<I{typ_len}sQH', typ_len,
                         typ_bytes, time_stamp, len(datas))
    body = b''.join(struct.pack('<?i', flag, val) for flag, val in datas)
    return header + body


def deserialize(data: bytes) -> Tuple[str, int, Tuple[bool, int], List[Tuple[bool, int]]]:
    offset = 0
    typ_len = struct.unpack_from('<I', data, offset)[0]
    offset += 4
    typ = struct.unpack_from(f'<{typ_len}s', data, offset)[0].decode('utf-8')
    offset += typ_len
    time_stamp = struct.unpack_from('<Q', data, offset)[0]
    offset += 8
    count = struct.unpack_from('<H', data, offset)[0]
    offset += 2
    datas: List[Tuple[bool, int]] = []
    for _ in range(count):
        flag = struct.unpack_from('<?', data, offset)[0]
        offset += 1
        val = struct.unpack_from('<i', data, offset)[0]
        offset += 4
        datas.append((flag, val))
    return typ, time_stamp, datas[0], datas[1:]


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


class ContextNetExtender(Extender):
    _process_shared_link_pool: Dict[str, asyncio.Queue[Tuple[Frames, Any]]] = {}

    def __init__(self, ctx: Context) -> None:
        self.ctx = ctx
        self.ctx.try_connect = self.try_connect
        self.net_ready: asyncio.Event = asyncio.Event()
        self.ctx.append_handler("NetBuilder.build", self._build)
        self.ctx.append_handler("NetBuilder.pref", self._pref_server)
        self.ctx.append_handler("NetBuilder.ret", self._on_callback)
        self.ctx.append_handler("NetBuilder.pref_lock", self._pref_lock)
        self.prefing: str = ""

        self.dsts: Set[str] = set()
        self.srcs: Set[str] = set()
        self.unreachable: Set[str] = set()
        self._cbs: Dict[str, asyncio.Future[Any]] = {}
        self._tcp_server_port: int = 0
        self._tcp_server: Any = None
        self._tcp_prefs: Dict[str, LinkPerfDescribe] = {}
        self._pendings: Dict[str, List[Tuple[str, Any, bool]]] = {}
        self._connecting: Dict[str, asyncio.Future[bool]] = {}

        def _default_writer(dst: str, api: str, data: Any, not_exist_ok: bool):
            if dst in self.unreachable:
                assert not_exist_ok, f"establish connection {ctx.name}->{dst} is known unreachable"
                return
            if dst in self._pendings:
                self._pendings[dst].append((api, data, not_exist_ok))
                return
            else:
                self._pendings[dst] = [(api, data, not_exist_ok)]
            ctx.logger.warn(f"start lazy connection establish {ctx.name}->{dst}")

            async def coro():
                ok = await self.try_connect(dst)
                pds = self._pendings.pop(dst)
                for api, data, not_exits_ok in pds:
                    if ok:
                        ctx.send((dst, api, data), not_exits_ok)
                    else:
                        assert not_exits_ok, f"cannot establish connection {ctx.name}->{dst}"
            asyncio.create_task(coro())

        ctx.set_default_writer(_default_writer)

        def pref_releaser(packet: Packet):
            (_, action, data) = packet
            if action == PacketTypeNodeDrop:
                if data == self.prefing:
                    self.prefing = ""

        self.ctx.append_listener(pref_releaser)

    def _pref_lock(self, packet: Packet):
        src, _, (identity_str, action) = packet
        if action == "acquire":
            if self.prefing != "":
                self.ctx.send((src, "NetBuilder.ret", (identity_str, False)))
            else:
                self.prefing = src
                self.ctx.send((src, "NetBuilder.ret", (identity_str, True)))
        if action == "release":
            if self.prefing == src:
                self.prefing = ""

    def _pref_server(self, packet: Packet):
        src, _, (identity_str, load) = packet
        recv_time = time.time_ns()
        self.ctx.send((src, "NetBuilder.ret",
                      (identity_str, (load, recv_time))))

    def _on_callback(self, packet: Packet):
        _, _, (identity_str, data) = packet
        w = self._cbs.pop(identity_str)
        w.set_result(data)

    def _make_callback(self, T: Type[_Ret]) -> Tuple[str, asyncio.Future[_Ret]]:
        id_str = uuid4().hex
        fut: asyncio.Future[_Ret] = asyncio.Future()
        self._cbs[id_str] = fut
        return id_str, fut

    @staticmethod
    def _packet_to_frames(typ: str, data: Any) -> Tuple[List[Union[bytes, memoryview]], Optional[Any]]:
        timestamp: int = time.time_ns()
        head_data = (False, -1)
        head_datas: List[Tuple[bool, int]] = []
        frames: List[Union[bytes, memoryview]] = [b'']
        no_gcs: List[Any] = []
        if isinstance(data, (bytes, memoryview)):
            # is bytes
            head_data = (False, len(data))
            frames.append(data)
        # elif isinstance(data, FasterSerializeObject):
        #     # is FasterSerializeObject
        #     hdr, more_frames, no_gc = data.to_faster_serialize_format()
        #     if no_gc is not None:
        #         no_gcs.append(no_gc)
        #     sd = pickle.dumps(hdr, protocol=-1)
        #     frames.append(sd)
        #     head_data = (True, len(sd))
        #     frames += more_frames
        #     for f in more_frames:
        #         head_datas.append((False, len(f)))
        elif isinstance(data, (list, tuple)):
            # is List[bytes|FasterSerializeObject|Any]
            for elem in cast(Tuple[Any], data):
                if isinstance(elem, (bytes, memoryview)):
                    head_datas.append((False, len(elem)))
                    frames.append(elem)
                # elif isinstance(elem, FasterSerializeObject):
                #     elem, more_frames, no_gc = elem.to_faster_serialize_format()
                #     if no_gc is not None:
                #         no_gcs.append(no_gc)
                #     de = pickle.dumps(elem, protocol=-1)
                #     head_datas.append((True, len(de)))
                #     frames.append(de)
                #     frames += more_frames
                #     for f in more_frames:
                #         head_datas.append((False, len(f)))
                else:
                    _frames, no_gc = FasterPickler.dumps(elem)
                    if no_gc is not None:
                        no_gcs.append(no_gc)
                    for i, f in enumerate(_frames):
                        if i == 0:
                            head_datas.append((True, len(f)))
                        else:
                            head_datas.append((False, len(f)))
                        frames.append(f)
        else:
            # is Any
            _frames, no_gc = FasterPickler.dumps(data)
            if no_gc is not None:
                no_gcs.append(no_gc)
            for i, f in enumerate(_frames):
                if i == 0:
                    head_data = (True, len(f))
                else:
                    head_datas.append((False, len(f)))
                frames.append(f)
        head_bytes = serialize(typ, timestamp, head_data, head_datas)
        frames[0] = head_bytes
        if len(no_gcs) == 0:
            no_gcs = None  # type:ignore
        return frames, no_gcs

    @staticmethod
    def _frames_to_packet(src_name: str, frames: asyncio.Frames) -> Tuple[Packet, int, int]:
        head, data_part = frames[0], frames[1:]
        typ, timestamp, head_data, head_datas = deserialize(head)
        data_part = cast(List[Union[bytes, memoryview]], data_part)
        total_bytes = 0
        recover_data: Any = None
        if head_data != (False, -1):
            rec, total_bytes = head_data
            if rec:
                recover_data, _, _ = FasterUnpickler.loads(data_part, frames)
                # if isinstance(recover_data, FasterDeSerializeObject):
                #     recover_data, _ = recover_data.full_recover(data_part[1:], frames)
            else:
                recover_data = bytes(data_part[0])
        else:
            recover_data = []
            assert len(head_datas) == len(data_part)
            total_frames = len(head_datas)
            i = 0
            while i < total_frames:
                h, p = head_datas[i], data_part[i]
                i += 1
                rec, count = h
                total_bytes += count
                if rec:
                    d, _, skip = FasterUnpickler.loads(data_part[i-1:], frames)
                    # if isinstance(d, FasterDeSerializeObject):
                    #     d, skip = d.full_recover(data_part[i:], frames)
                    i += skip-1
                    recover_data.append(d)
                else:
                    recover_data.append(bytes(p))
        packet = (src_name, typ, recover_data)
        return packet, timestamp, total_bytes

    async def _tcp_handle_fn(self, conn: asyncio.FrameConn):
        src_name_bytes = await conn.recv()
        conn.send([b"ACK"], None)
        assert src_name_bytes is not None
        src_name = bytes(src_name_bytes[0]).decode()

        link_args = self._tcp_prefs[src_name]
        latency_limit = link_args.latency_ms/1000  # s
        bandwidth_limit: float = link_args.bandwidth_byte_s  # B/s
        loop = asyncio.get_event_loop()

        self.srcs.add(src_name)
        while True:
            frames = await conn.recv()
            if frames is None:
                break
            # return frames

            # frames = await fread.read_frames()
            packet, timestamp, total_bytes = self._frames_to_packet(
                src_name, frames)
            if latency_limit == 0 and bandwidth_limit == 0:
                self.ctx.put_packet(packet)
            else:
                expect_time_s = latency_limit+(total_bytes/bandwidth_limit)
                elaps_time_s = (time.time_ns()-timestamp)/(10**9)
                sleep_time_s = expect_time_s-elaps_time_s
                if sleep_time_s <= 0.001:
                    self.ctx.put_packet(packet)
                else:
                    loop.call_later(sleep_time_s, self.ctx.put_packet, packet)
        self.ctx.logger.warn(f"{src_name} dropped")
        self.srcs.remove(src_name)
        self.ctx.put_packet((OrbitName, PacketTypeNodeDrop, src_name))
        if src_name in self.dsts:
            self.dsts.remove(src_name)

    async def _get_tcp_open_port(self):
        # dst
        _network_hostname = self.ctx.network_hostname
        network_hostnames: List[str] = []
        if _network_hostname == "":
            network_hostnames = get_all_ips()
        elif "," in _network_hostname:
            network_hostnames = [s for s in _network_hostname.split(",")]
        else:
            network_hostnames = [_network_hostname]
        if self._tcp_server_port != 0:
            return (network_hostnames, self._tcp_server_port)
        else:
            self._tcp_server_port = get_free_port()
            self._tcp_server = await asyncio.get_event_loop().tcp_server(f"0.0.0.0:{self._tcp_server_port}", self._tcp_handle_fn)
            return (network_hostnames, self._tcp_server_port)

    async def _establish_connect(self, dst_name: str, hosts: List[str], port: int):
        # src
        self.ctx.logger.info(f"establishing connect to {dst_name} through  [{', '.join(hosts)}]:{port}")
        conn: Optional[FrameConn] = None
        for host in hosts:
            try:
                conn = await asyncio.get_event_loop().tcp_connect(f"{host}:{port}")
                self.ctx.logger.info(f"established connect: {host}:{port}")
                break
            except Exception as e:
                self.ctx.logger.warn(f"cannot establish connection through: {host}:{port}")
                conn = None
        if conn is None:
            raise ValueError(f"fail to connect {dst_name} through [{', '.join(hosts)}]:{port}")
        self.dsts.add(dst_name)
        # src send name
        conn.send([self.ctx.location.name.encode()], None)
        ack = await conn.recv()
        assert ack is not None and bytes(ack[0]) == b"ACK"

        def writer(typ: str, data: Any):
            frames, no_gcs = self._packet_to_frames(typ, data)
            # if no_gcs is None:
            #     conn.send(frames, None)
            # else:

            def _on_release():
                nonlocal no_gcs
                nonlocal data
                del no_gcs
                del data
            conn.send(frames, _on_release)
        self.ctx.add_writer(dst_name, writer)
        self.unreachable.discard(dst_name)

    def _virtual_net_dst(self, src_name: str, dst_name: str, link_args: LinkPerfDescribe):
        link_name = f"{src_name}->{dst_name}"
        q: asyncio.Queue[Tuple[Frames, Any]] = asyncio.Queue()
        latency_limit = link_args.latency_ms/1000  # s
        bandwidth_limit = link_args.bandwidth_byte_s  # B/s
        loop = asyncio.get_event_loop()
        assert dst_name == self.ctx.location.name
        self.srcs.add(src_name)
        self._process_shared_link_pool[link_name] = q

        async def recv_coro():
            while True:
                frames, no_gcs = await q.get()
                packet, timestamp, total_bytes = self._frames_to_packet(
                    src_name, frames)
                if latency_limit == 0 and bandwidth_limit == 0:
                    self.ctx.put_packet(packet)
                else:
                    expect_time_s = latency_limit+(total_bytes/bandwidth_limit)
                    elaps_time_s = (time.time_ns()-timestamp)/(10**9)
                    sleep_time_s = expect_time_s-elaps_time_s
                    if sleep_time_s <= 0.001:
                        self.ctx.put_packet(packet)
                    else:
                        loop.call_later(sleep_time_s, self.ctx.put_packet, packet)
                del no_gcs
        self.ctx.append_no_gc(asyncio.create_task(recv_coro()))

    def _virtual_net_src(self, src_name: str, dst_name: str):
        link_name = f"{src_name}->{dst_name}"
        self.dsts.add(dst_name)
        assert src_name == self.ctx.location.name
        q = self._process_shared_link_pool[link_name]

        def writer(typ: str, data: Any):
            frames, no_gcs = self._packet_to_frames(typ, data)
            q.put_nowait((frames, no_gcs))
        self.ctx.add_writer(dst_name, writer)

    def _build(self, packet: Packet):
        resp_uuid, peer_name, role, link_type, link_args = packet[2]
        assert link_type in ("virtual", "tcp")
        assert role in ("reciver", "sender")
        if link_type == "virtual":
            if role == "reciver":
                self._virtual_net_dst(peer_name, self.ctx.location.name, cast(LinkPerfDescribe, link_args))
            else:
                self._virtual_net_src(self.ctx.location.name, peer_name)
            self.ctx.send((OrbitName, "NetBuilder.resp", (resp_uuid, None)))
        elif link_type == "tcp":
            if role == "reciver":
                self._tcp_prefs[peer_name] = link_args

                async def _tcp_dst():
                    dst_addr = await self._get_tcp_open_port()
                    self.ctx.send((OrbitName, "NetBuilder.resp", (resp_uuid, dst_addr)))
                asyncio.create_task(_tcp_dst())
            else:
                async def _tcp_src():
                    (hosts, port) = link_args
                    await self._establish_connect(peer_name, hosts, port)
                    self.ctx.send(
                        (OrbitName, "NetBuilder.resp", (resp_uuid, None)))
                asyncio.create_task(_tcp_src())

    # async def wait_node_net_ready(self) -> None:
    #     self.ctx.send((OrbitName, "NetBuilder.init", self.ctx.location))
    #     await self.net_ready.wait()

    def connected(self, dst: str):
        return dst in self.dsts

    async def try_connect(self, target: str) -> bool:
        if self.connected(target):
            return True
        if target in self._connecting:
            fut = self._connecting[target]
        else:
            id_str, fut = self._make_callback(bool)
            self._connecting[target] = fut
            self.ctx.send((OrbitName, "NetBuilder.connect", (id_str, target)))
        ok = await fut
        self._connecting.pop(target)
        if not ok:
            self.unreachable.add(target)
        return ok

    async def pref_net(self, target_node: LocationByName, repeat_times: int = 5) -> Tuple[SendPerf, RecvPref]:
        async def do_pref(size: int):
            """
            return this->remote time, remote->this time
            """
            ident, fut = self._make_callback(Tuple[any, int])
            load = b' ' * size
            self.ctx.send((target_node, "NetBuilder.pref", (ident, load)))
            _, recv_ts = await fut
            current_ts = time.time_ns()
            return recv_ts, current_ts
        sizes: List[int] = [100, 500, 2000, 10000]

        U: List[Tuple[int, float]] = []  # (size, U_val)
        D: List[Tuple[int, float]] = []  # (size, D_val)
        RTTs: List[float] = []

        while True:
            id_str, fut = self._make_callback(bool)
            self.ctx.send((target_node, "NetBuilder.pref_lock", (id_str, "acquire")))
            if not (await fut):
                self.ctx.logger.info(f"{target_node} pref server busy, waiting...")
                await asyncio.sleep(1)
            else:
                break

        for size in sizes:
            for _ in range(repeat_times):
                t0 = time.time_ns()
                t1r, t2 = await do_pref(size)
                RTTs.append((t2 - t0)/(10**9))
                U.append((size, (t1r - t0)/(10**9)))   # raw U(s)
                D.append((size, (t2 - t1r)/(10**9)))   # raw D(s)

        self.ctx.send((target_node, "NetBuilder.pref_lock", ("", "release")))

        sizes_u = np.array([s for s, _ in U]).reshape(-1, 1)
        vals_u = np.array([v for _, v in U])

        sizes_d = np.array([s for s, _ in D]).reshape(-1, 1)
        vals_d = np.array([v for _, v in D])

        from sklearn.linear_model import RANSACRegressor, LinearRegression
        # RANSAC U(s) = A + B*s
        ransac_u = RANSACRegressor(LinearRegression(), min_samples=0.7)
        ransac_u.fit(sizes_u, vals_u)
        slope_up = ransac_u.estimator_.coef_[0]
        intercept_up = ransac_u.estimator_.intercept_

        # RANSAC D(s) = C + D*s
        ransac_d = RANSACRegressor(LinearRegression(), min_samples=0.7)
        ransac_d.fit(sizes_d, vals_d)
        slope_down = ransac_d.estimator_.coef_[0]
        intercept_down = ransac_d.estimator_.intercept_
        # slope_up   = drift / bw_up
        # slope_down = 1 / (drift * bw_down)
        # min RTT    = latency_up + latency_down + size/bw_up + size/bw_down (size=0)
        rtt_min = min(RTTs)

        # latency_up = intercept_up  - offset*(drift-1)
        # latency_down = intercept_down - offset*(1/drift -1)
        # latency_up + latency_down = rtt_min
        # =>  (intercept_up + intercept_down)  - offset*(drift - 1/drift) = rtt_min

        def equation(drift: float):
            if drift <= 0:
                return 1e9
            return (intercept_up + intercept_down) - rtt_min - 0.000001 * (drift - 1/drift)
        scan = np.linspace(0.95, 1.05, 2001)
        values = np.abs([equation(d) for d in scan])
        drift_best = scan[np.argmin(values)]

        drift = drift_best
        bw_up = drift / slope_up
        if bw_up < 0 or bw_up > (100*2**30):
            bw_up = 100*2**30
        bw_down = 1 / (drift * slope_down)
        if bw_down < 0 or bw_down > (100*2**30):
            bw_down = 100*2**30
        # latency_up_from_min = min(U(s) - s/bw_up)
        # latency_down_from_min = min(D(s) - s/bw_down)
        latency_up = max(0, min(v - s / bw_up for s, v in U))
        latency_down = max(0, min(v - s / bw_down for s, v in D))

        return SendPerf(latency_up*1000, bw_up/(2**(20-3))), RecvPref(latency_down*1000, bw_down/(2**(20-3)))
