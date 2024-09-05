import argparse
import datetime
import logging
import os
import re
import subprocess
from pathlib import Path

from common import multi_thread_launcher
from pydantic import BaseModel
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn

parser = argparse.ArgumentParser(
    prog="Kita",
    description="Kita is a compiler performance testing and analysis tool.",
    epilog="Notice: Kita requires your device to have qemu installed to run the test or you have the native environment."
    + "Please prepare `libsysy_riscv.a`(RISCV64) or `sylib.a`(ARM) at current directory if you are not using native environment.",
)

parser.add_argument("samples", type=str, help="Path to the samples directory.")
parser.add_argument("compiler", type=str, help="Path to the compiler.")
parser.add_argument(
    "--args", nargs=argparse.REMAINDER, help="Arguments for the compiler."
)
parser.add_argument(
    "--strict", action="store_true", help="Terminate test when any checkpoint fail."
)
parser.add_argument(
    "-M", "--multi-thread", action="store_true", help="Use multi-threading for tasks."
)
parser.add_argument(
    "-C",
    "--contrast",
    type=str,
    help="Path to the contrast data file, default is last test result.",
)
parser.add_argument(
    "-N",
    "--native",
    action="store_true",
    help="Use native environment as testing environment.",
)
parser.add_argument("-S", "--silent", action="store_true", help="Hide compiler output.")
parser.add_argument("--keep", action="store_true", help="Skip the cleanup process.")
parser.add_argument(
    "-O",
    "--output",
    type=str,
    dest="report_path",
    help="Path to the output report directory, default is ./report.",
    default="./report",
)
parser.add_argument(
    "-P",
    "--platform",
    type=str,
    dest="platform",
    choices=["arm", "riscv64"],
    help="Specific the target platform, default is riscv64.",
    default="riscv64",
)
parser.add_argument(
    "-L",
    "--level",
    type=str,
    dest="log_level",
    help="Specific the log level, default is INFO.",
    choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    default="INFO",
)


class ContrastData(BaseModel):
    compile: datetime.timedelta
    is_multi_thread: bool = False
    test: dict[str, str]


args = parser.parse_args()
console = Console(record=True)
report_path = Path(args.report_path)
logging.basicConfig(
    level=args.log_level,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True, console=console, markup=True)],
)
logger = logging.getLogger("rich")

if not report_path.exists():
    os.makedirs(report_path)

if args.contrast is None or not (sp_contrast := Path(args.contrast)).exists():
    report = list(report_path.glob("*.json"))
    if report.__len__() == 0:
        sp_contrast = None
    else:
        report.sort()
        sp_contrast = report[-1]

contrast_data = None
current_data = ContrastData(compile=datetime.timedelta(), test={})
if sp_contrast is not None:
    logger.info(
        f"[green] Loading contrast data from {sp_contrast.absolute()}...[/green]"
    )
    contrast_data = ContrastData.model_validate_json(sp_contrast.read_text())
else:
    logger.info("[yellow]No contrast data found.[/yellow]")

if not os.path.exists(args.compiler):
    logger.error(f"[red]Error: The compiler does not exist at {args.compiler}[/red]")
    exit(1)

os.chmod(Path(args.compiler).absolute().__str__(), 0o777)

if not os.path.exists(args.samples):
    logger.error(
        f"[red]Error: The samples directory does not exist at {args.samples}[/red]"
    )
    exit(1)

samples = [
    Path(f"{args.samples}/{f}")
    for f in os.listdir(Path(args.samples))
    if f.endswith(".sy") and os.path.isfile(f"{args.samples}/{f}")
]

filtered_samples: list[Path] = []

for sample in samples:
    if not sample.exists() or not os.path.exists(sample.with_suffix(".out")):
        if args.strict:
            logger.error(
                f"[red]Error: The [yellow]{sample}[/yellow] is not a valid checkpoint.[/red]"
            )
            exit(1)
        else:
            logger.warning(
                f"[yellow]Warning: The [blue]{sample}[/blue] is not a valid checkpoint.[/yellow]"
            )
    else:
        filtered_samples.append(sample)

logger.info(
    f"[green]Samples Directory: [yellow]{Path(args.samples).absolute()}[/yellow][/green]"
)
if filtered_samples.__len__() == 0:
    logger.error(
        f"[red]Error: The samples directory does not contain any valid checkpoints.[/red]"
    )
    exit(1)
logger.info(f"[green]Found [yellow]{samples.__len__()}[/yellow] sample(s).[/green]")

if not args.keep:
    logger.info("[blue]Cleaning up output directory...[/blue]")
    for sample in filtered_samples:
        sample.with_suffix(".s").unlink(True)
        sample.with_suffix(".target").unlink(True)
        sample.with_suffix(".tst").unlink(True)
    logger.info("[blue]Cleanup finished.[/blue]")

fail_cnt = 0

with Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    console=console,
    transient=True,
    expand=True,
) as progress:
    start_compile_time = datetime.datetime.now()
    compile_task = progress.add_task("Compiling...", total=len(filtered_samples))

    def checkpoint_1(sample: Path):
        logger.info(f"[yellow]Compiling {sample.stem}[/yellow]")
        compile_args = [
            Path(args.compiler).absolute().__str__(),
            "-S",
            "-o",
            sample.with_suffix(".s").absolute().__str__(),
            sample.absolute().__str__(),
        ]
        if args.args:
            compile_args.extend(args.args)
        try:
            result = subprocess.run(
                compile_args,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            msg = result.stdout.decode("utf-8").strip()
            error = result.stderr.decode("utf-8").strip()
            if msg.find("error") != -1 or msg.find("err") != -1:
                if not args.silent and error.__len__() != 0:
                    logger.error(error)
                raise Exception("Compile Error.")
            if not args.silent and msg.__len__() != 0:
                logger.info(msg)
            if result.returncode != 0:
                if not args.silent and error.__len__() != 0:
                    logger.error(error)
                raise Exception("Compile Error.")
            if not sample.with_suffix(".s").exists():
                raise Exception("No output file generated.")
            logger.info(f"[green]Compiled {sample.stem} successfully.[/green]")
            progress.advance(compile_task)
        except Exception as e:
            global fail_cnt
            fail_cnt += 1
            logger.error(f"[red]Error: Failed to compile {sample.stem}.[/red]")
            if not args.silent:
                logger.exception(e)
            if args.strict:
                exit(1)

    if args.multi_thread:
        current_data.is_multi_thread = True
        multi_thread_launcher(filtered_samples, checkpoint_1)
    else:
        for sample in filtered_samples:
            checkpoint_1(sample)
    compile_used_time = datetime.datetime.now() - start_compile_time

    logger.info(
        f"\n\n[bold cyan]Compiled [blue]{filtered_samples.__len__()}[/blue] file(s)"
        + "(Multi-thread)"
        if current_data.is_multi_thread
        else ""
        + f"([green]{filtered_samples.__len__()-fail_cnt} Succeed[/green], [red]{fail_cnt} Failed[/red]) "
        + f"in [blue]{compile_used_time}[/blue].\n\n"
    )

    if contrast_data is not None:
        delta_time = compile_used_time - contrast_data.compile
        logger.info(
            f"[green]Previous result: {contrast_data.compile}[/green]"
            + f"[bold red] + {compile_used_time-contrast_data.compile}"
            if delta_time.total_seconds() > 0
            else f"[bold green] - {-delta_time}"
        )
    if fail_cnt > 0:
        logger.warning(
            f"[bold yellow]{fail_cnt} sample(s) failed to compile. [/bold yellow]"
        )
    fail_cnt = 0

    current_data.compile = compile_used_time
    progress.remove_task(compile_task)

    target_task = progress.add_task("Targeting...", total=len(filtered_samples))

    start_target_time = datetime.datetime.now()

    def checkpoint_2(sample: Path):
        logger.info(f"[yellow]Targeting {sample.stem}[/yellow]")
        target_args = []
        if args.native:
            target_args.extend(
                [
                    "gcc",
                    "-o",
                    sample.with_suffix(".target").absolute().__str__(),
                    sample.with_suffix(".s").absolute().__str__(),
                ]
            )
        elif args.platform == "arm":
            target_args.extend(
                [
                    "arm-linux-gnueabihf-gcc",
                    "-o",
                    sample.with_suffix(".target").absolute().__str__(),
                    sample.with_suffix(".s").absolute().__str__(),
                    Path("./sylib.a").absolute().__str__(),
                ]
            )
        elif args.platform == "riscv64":
            target_args.extend(
                [
                    "riscv64-linux-gnu-gcc",
                    "-march=rv64gc",
                    "-o",
                    sample.with_suffix(".target").absolute().__str__(),
                    sample.with_suffix(".s").absolute().__str__(),
                    Path("./libsysy_riscv.a").absolute().__str__(),
                ]
            )
        try:
            result = subprocess.run(
                target_args,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            msg = result.stdout.decode("utf-8").strip()
            error = result.stderr.decode("utf-8").strip()
            if msg.find("error") != -1 or msg.find("err") != -1:
                if not args.silent and error.__len__() != 0:
                    logger.error(error)
                raise Exception("Target Error.")
            if not args.silent and msg.__len__() != 0:
                logger.info(msg)
            if result.returncode != 0:
                if not args.silent and error.__len__() != 0:
                    logger.error(error)
                raise Exception("Target Error.")
            if not sample.with_suffix(".target").exists():
                raise Exception("No output file generated.")
            logger.info(f"[green]Targeted {sample.stem} successfully.\n[/green]")
            os.chmod(sample.with_suffix(".target").absolute().__str__(), 0o777)
            progress.advance(target_task)
        except Exception as e:
            global fail_cnt
            fail_cnt += 1
            logger.error(f"[red]Error: Failed to target {sample.stem}.[/red]")
            if not args.silent:
                logger.exception(e)
            if args.strict:
                exit(1)

    if args.multi_thread:
        multi_thread_launcher(filtered_samples, checkpoint_2)
    else:
        for sample in filtered_samples:
            checkpoint_2(sample)

    logger.info(
        f"\n\n[bold cyan]Targeted [blue]{filtered_samples.__len__()}[/blue] file(s) "
        + f"([green]{filtered_samples.__len__()-fail_cnt} Succeed[/green], [red]{fail_cnt} Failed[/red]) "
        + f"in [blue]{datetime.datetime.now() - start_target_time}[/blue].[/bold cyan]\n\n"
    )
    if fail_cnt > 0:
        logger.warning(
            f"[bold yellow]{fail_cnt} sample(s) failed to target. [/bold yellow]"
        )

    fail_cnt = 0
    progress.remove_task(target_task)

    test_task = progress.add_task("Testing...", total=len(filtered_samples))

    def checkpoint_3(sample: Path):
        global fail_cnt
        logger.info(f"[yellow]Testing {sample.stem}[/yellow]")
        test_args = []

        if args.native:
            test_args.extend([sample.with_suffix(".target").absolute().__str__()])
        elif args.platform == "arm":
            test_args.extend(
                [
                    "qemu-arm",
                    "-L",
                    "/usr/arm-linux-gnueabihf",
                    sample.with_suffix(".target").absolute().__str__(),
                ]
            )
        elif args.platform == "riscv64":
            test_args.extend(
                [
                    "qemu-riscv64",
                    "-L",
                    "/usr/riscv64-linux-gnu",
                    sample.with_suffix(".target").absolute().__str__(),
                ]
            )

        try:
            output = open(cmd_output := (sample.with_suffix(".tst")), "w")
            if sample.with_suffix(".in").exists():
                result = subprocess.run(
                    test_args,
                    stdout=output,
                    stderr=subprocess.PIPE,
                    input=sample.with_suffix(".in").read_bytes(),
                )
            else:
                result = subprocess.run(
                    test_args,
                    stdout=output,
                    stderr=subprocess.PIPE,
                )
            output.write(f"{result.returncode}")
            output.close()
            if not sample.with_suffix(".tst").exists():
                raise Exception("No output file generated.")
            err = result.stderr.decode("utf-8").strip()
            match = re.search(r"(\d+)H-(\d+)M-(\d+)S-(\d+)us", err)
            if match:
                used_time = datetime.timedelta(
                    hours=int(match[1]),
                    minutes=int(match[2]),
                    seconds=int(match[3]),
                    microseconds=int(match[4]),
                )
                logger.info(f"[green]Used time: [blue]{used_time}[blue][/green]")
                if contrast_data is not None:
                    if sample.stem in contrast_data.test:
                        delta_time = used_time - datetime.timedelta(
                            seconds=float(
                                contrast_data.test[sample.stem].replace("s", "")
                            )
                        )
                        logger.info(
                            f"[green]Previous result: {contrast_data.test[sample.stem]}[/green]"
                            + f"[bold red] + {delta_time}[/bold red]"
                            if delta_time.total_seconds() > 0
                            else f"[bold green] - {-delta_time}[/bold green]"
                        )
                    else:
                        logger.debug(
                            "[green]This checkpoint does not have time information.[/green]"
                        )
                    current_data.test[sample.stem] = f"{used_time.total_seconds()}s"
            else:
                logger.debug(
                    "[green]This checkpoint does not have time information.[/green]"
                )

            if (ret := cmd_output.read_text().replace("\n", "")) != (
                src := sample.with_suffix(".out").read_text().replace("\n", "")
            ):
                raise Exception(f"Expected:\n {src}\n,  but found:\n {ret}")
            else:
                logger.info(f"[green]Test {sample.stem} passed.[/green]")
                progress.advance(test_task)
            progress.advance(test_task)
        except Exception as e:
            fail_cnt += 1
            logger.error(f"[red]Error: Failed to test {sample.stem}.[/red]")
            if not args.silent:
                logger.exception(e)
            if args.strict:
                exit(1)

    for sample in filtered_samples:
        checkpoint_3(sample)
    logger.info("[bold cyan]Test finished.[/bold cyan]")
    if fail_cnt > 0:
        logger.warning(
            f"[bold yellow]{fail_cnt} sample(s) failed to test. [/bold yellow]"
        )

    logger.info("[bold cyan]Generating report[/bold cyan]")
    with open(
        report_path.absolute()
        .joinpath(f"{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json")
        .__str__(),
        "x",
    ) as f:
        f.write(current_data.model_dump_json())

console.save_html(
    report_path.absolute()
    .joinpath(f"report_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.html")
    .absolute()
    .__str__()
)

logger.info(
    "[bold cyan]Report generated at [/bold cyan]" + report_path.absolute().__str__()
)
