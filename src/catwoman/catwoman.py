import argparse
import logging
import sys

from catwoman import __version__

__author__ = "Lisa McBride"
__copyright__ = "Lisa McBride"
__license__ = "MIT"

_logger = logging.getLogger(__name__)

def read_cube(path, type=np.float64):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    try :
        cube = np.fromfile(path, dtype=type)
    except FileNotFoundError :
        print(" !!!!!! file not found : "+ path)
        cube = np.zeros(256**3)
        print("moving on...")
    shape = np.shape(cube)[0]
    length = int(shape**(1/3)) +1

    cube =np.reshape(cube, (length,length,length)).T
    shape = np.shape(cube)

    return cube

def fetch_spectra(file_n, sim_n=10038):
    """Fibonacci example function

    Args:
      n (int): integer

    Returns:
      int: n-th Fibonacci number
    """
    print(f'Now parsing /Users/emcbride/kSZ/data/Pee_spectra_LoReLi/raw/simu{sim_n}/')
    z_fn = f'/Users/emcbride/kSZ/data/Pee_spectra_LoReLi/raw/simu{sim_n}/redshift_list.dat'
    redshifts = {}

    with open(z_fn) as f:
        for line in f:
           (val, key) = line.split()
           redshifts[key] = val

    #print(redshifts)

    Pee_list = []
    for n in file_n:
        print(f'Now on file {n}')

        ion_file = f'/Users/emcbride/kSZ/data/xion/xion_256_out{n}.dat'
        ion_cube = read_cube(ion_file)

        dens_file = f'/Users/emcbride/kSZ/data/dens/dens_256_out{n}.dat'
        dens_cube = read_cube(dens_file)

        Pee_file = f'/Users/emcbride/kSZ/data/Pee_spectra_LoReLi/formatted/simu{sim_n}/postprocessing/cubes/ps_dtb/powerspectrum_electrons{n}.dat'
        P_ee = np.loadtxt(Pee_file).T
        z = redshifts[n]

        spectra_dict = {'file_n': n,
                        'z': float(z),
                        'dens_cube': dens_cube,
                        'ion_cube': ion_cube,
                        'k': P_ee[0],
                        'P_k': P_ee[1]}
        Pee_list.append(spectra_dict)

    return Pee_list


# ---- CLI ----
# The functions defined in this section are wrappers around the main Python
# API allowing them to be called directly from the terminal as a CLI
# executable/script.


def parse_args(args):
    """Parse command line parameters

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--help"]``).

    Returns:
      :obj:`argparse.Namespace`: command line parameters namespace
    """
    parser = argparse.ArgumentParser(description="Just a Fibonacci demonstration")
    parser.add_argument(
        "--version",
        action="version",
        version=f"catwoman {__version__}",
    )
    parser.add_argument(dest="n", help="n-th Fibonacci number", type=int, metavar="INT")
    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    return parser.parse_args(args)


def setup_logging(loglevel):
    """Setup basic logging

    Args:
      loglevel (int): minimum loglevel for emitting messages
    """
    logformat = "[%(asctime)s] %(levelname)s:%(name)s:%(message)s"
    logging.basicConfig(
        level=loglevel, stream=sys.stdout, format=logformat, datefmt="%Y-%m-%d %H:%M:%S"
    )


def main(args):
    """Wrapper allowing :func:`fib` to be called with string arguments in a CLI fashion

    Instead of returning the value from :func:`fib`, it prints the result to the
    ``stdout`` in a nicely formatted message.

    Args:
      args (List[str]): command line parameters as list of strings
          (for example  ``["--verbose", "42"]``).
    """
    args = parse_args(args)
    setup_logging(args.loglevel)
    _logger.debug("Starting crazy calculations...")
    print(f"The {args.n}-th Fibonacci number is {fib(args.n)}")
    _logger.info("Script ends here")


def run():
    """Calls :func:`main` passing the CLI arguments extracted from :obj:`sys.argv`

    This function can be used as entry point to create console scripts with setuptools.
    """
    main(sys.argv[1:])


if __name__ == "__main__":
    # ^  This is a guard statement that will prevent the following code from
    #    being executed in the case someone imports this file instead of
    #    executing it as a script.
    #    https://docs.python.org/3/library/__main__.html

    # After installing your project with pip, users can also run your Python
    # modules as scripts via the ``-m`` flag, as defined in PEP 338::
    #
    #     python -m catwoman.skeleton 42
    #
    run()
