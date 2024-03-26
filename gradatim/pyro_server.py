
import Pyro4
import Pyro4.naming
import threading
from gradatim.dosa import dosa, DosaModelType


@Pyro4.expose
class DOSAFwd(object):
    def dosa_fwd(self, dosa_config_path, model_type: str, model_path: str, const_path: str, global_build_dir: str,
                 show_graphics: bool = True, generate_build: bool = True, generate_only_stats: bool = False,
                 generate_only_coverage: bool = False, calibration_data: str = None, map_weights_path: str = None):
        a_model_type = DosaModelType.ONNX
        if model_type == 'torchscript':
            a_model_type = DosaModelType.TORCHSCRIPT
        return dosa(dosa_config_path, a_model_type, model_path, const_path, global_build_dir, show_graphics,
                    generate_build, generate_only_stats, generate_only_coverage, calibration_data, map_weights_path)


def pyro_server():
    daemon = Pyro4.Daemon()
    uri = daemon.register(DOSAFwd)
    print(f"Pyro-4-uri: {uri}")
    background_thread = threading.Thread(target=daemon.requestLoop)
    background_thread.start()
    print("Pyro4 background process started...\n")
    # wait forever...
    background_thread.join(timeout=None)


if __name__ == '__main__':
    pyro_server()
