import logging
import os
import time
import traceback
from pprint import pprint

import click
import dtadeviceservice
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from click_repl import repl
from click_repl.exceptions import ExitReplException
from prompt_toolkit.history import FileHistory

from vanya import logger, setup_logger
from vanya.devices.firestick import provision_firestick, FIRESTICK_APK, FIRESTICK_DATA
from vanya.nrdp import NRDP
from vanya.utils.version import get_version

# device = DeviceService()

CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"])
HISTORY_FILE = os.environ["HOME"] + "/.vanya-history"


# noinspection PyUnusedLocal
def exit_on_non_confirmation_callback(ctx, param, value):
    """Callback routine for click."""
    if not value:
        ctx.exit()


@click.group(invoke_without_command=True, context_settings=CONTEXT_SETTINGS)
@click.option(
    "--debug", "-d", default=False, is_flag=True, help="Turn on debug messages."
)
@click.pass_context
def cli(ctx, debug):
    """CLI and Python module for managing devices in the Netflix labs."""
    setup_logger(debug)
    logger.debug("Logging enabled ....")
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "w+") as file:
            file.write("")
    if ctx.invoked_subcommand is None:
        # noinspection PyBroadException
        try:
            ctx.invoke(command_repl)
        except ExitReplException:
            exit()
        except Exception:
            traceback.print_exc()


# noinspection PyUnusedLocal
@click.command(name="exit")
@click.pass_context
def command_exit(ctx):
    """Exit vanya (aliases: exit, e, quit, q)"""
    raise ExitReplException


# noinspection PyUnusedLocal
@click.command(name="e", hidden=True)
@click.pass_context
def command_e(ctx):
    raise ExitReplException


# noinspection PyUnusedLocal
@click.command(name="quit", hidden=True)
@click.pass_context
def command_quit(ctx):
    raise ExitReplException


# noinspection PyUnusedLocal
@click.command(name="q", hidden=True)
@click.pass_context
def command_q(ctx):
    raise ExitReplException


@click.command(name="help")
@click.argument("topic", default=None, required=False, nargs=1)
@click.pass_context
def command_help(ctx, topic):
    """Help command"""
    if topic is None:
        click.echo(ctx.parent.get_help())
    else:
        if topic in cli.commands:
            click.echo(cli.commands[topic].get_help(ctx))
        else:
            click.echo("Unknown command: ", nl=False)
            click.secho("{}\n".format(topic), bold=True, fg="red")
            click.echo(ctx.parent.get_help())


# noinspection PyUnusedLocal
@click.command(name='get-esn')
@click.option('--ip-address', '-i', default="127.0.0.1", help='IP address of NRDP application.')
@click.option('--port', '-p', default=9536, help='NRDP console port.')
@click.option('--debug', '-d', default=False, is_flag=True, help='Turn on debug messages.')
def command_get_esn(ip_address, port, debug):
    """Get the ESN of the NRDP application instance."""
    try:
        nrdp = NRDP(ip_address, port)
        esn = nrdp.get_esn()
        click.echo("NRDP app on {} ESN {}".format(ip_address, esn))
    except RuntimeError as err:
        click.echo(str(err), color='red', err=True)
        logging.exception("Problem getting ESN for NRDP app on {}".format(ip_address))


@click.command(name="repl", hidden=True)
def command_repl():
    """Start vanya in a read-eval-print-loop (REPL) interactive shell."""
    print("Vanya v{}".format(get_version()))
    prompt_kwargs = {
        "message": u"vanya> ",
        "history": FileHistory(HISTORY_FILE),
    }
    try:
        repl(click.get_current_context(), prompt_kwargs=prompt_kwargs)
    except ExitReplException:
        exit()


@click.command(name="version")
def command_version():
    """Display vanya version information."""
    logger.debug("In version .... DEBUG message")
    logger.info("In version ... INFO message")
    print("Vanya v{}".format(get_version()))


# noinspection PyUnusedLocal
@click.command(name='provision-firestick')
@click.option('--build', '-b', default=FIRESTICK_APK,
              help='Build file to install')
@click.option('--debug', '-d', default=False, is_flag=True, help='Turn on debug messages.')
@click.option('--ip-address', '-i', default="127.0.0.1", help='IP address of NRDP application.')
@click.option('--label', '-l', default="", help='Labe/ID of Firestickl.')
@click.option('--ip-address-file', '-f', default=None, help='File of IP addresses for devices to provisions.')
@click.option('--provisioning-data-file', '-p', default=FIRESTICK_DATA,
              help='File to store device provisioning information. ')
@click.option('--update-device-map', '-m', default=False, is_flag=True,
              help='File of IP addresses for devices to provisions.')
def command_provision_firestick(ip_address, label, ip_address_file, provisioning_data_file, build, update_device_map,
                                debug):
    """Provisioning Amazon Firestick
    """
    device_service = dtadeviceservice.DefaultApi()
    if ip_address_file is not None:
        with open(ip_address_file) as ip_addresses:
            devices = ip_addresses.read().splitlines()
            for device in devices:
                (address, label) = device.split(",")
                provision_firestick(address, label, build, device_service, update_device_map, provisioning_data_file)
                time.sleep(3)
    else:
        provision_firestick(ip_address, label, build, device_service, update_device_map, provisioning_data_file)


# noinspection PyUnusedLocal
@click.command(name='update-device-map')
@click.option('--debug', '-d', default=False, is_flag=True, help='Turn on debug messages.')
@click.option('--parameters-filename', '-p', default=None, help='YAML file with default parameters.')
@click.option('--devices-filename', '-f', default=None, help='CSV file of device info.')
def command_update_device_map(parameters_filename, devices_filename, debug):
    """Bulk update device map
    """
    device_service = dtadeviceservice.DefaultApi()
    if devices_filename is not None and parameters_filename is not None:
        with open(parameters_filename) as default_parameters_file:
            default_parameters = load(default_parameters_file, Loader=Loader)
        with open(devices_filename) as devices_file:
            devices = devices_file.read().splitlines()
            for device in devices:
                (label, esn, host, mac) = device.split(",")
                device_info = default_parameters
                device_info.update({'host': host,
                                    'mac': mac})
            try:
                print("****** Add device info to device map for ESN :\n {}".format(esn))
                pprint(device_info)
                response = device_service.delete_device_map(esn)
                response = device_service.update_device_map(device_info, esn)
                pprint(response)

            except Exception as e:
                print("Exception when calling DefaultApi->install: %s\n" % e)

            try:
                # Given an ESN, return the Device Map Entry
                print("\n\n\n\nGetting the device map ...")
                response = device_service.get_device_map(esn)
                pprint(response)
            except Exception as e:
                print("Exception when calling DefaultApi->get_device_map: %s\n" % e)


cli.add_command(command_e)
cli.add_command(command_exit)
cli.add_command(command_q)
cli.add_command(command_quit)
cli.add_command(command_help)
cli.add_command(command_get_esn)
cli.add_command(command_repl)
cli.add_command(command_version)
cli.add_command(command_provision_firestick)
cli.add_command(command_update_device_map)
