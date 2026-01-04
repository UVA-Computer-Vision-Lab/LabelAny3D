# Copyright (c) Meta Platforms, Inc. and affiliates
from termcolor import colored
import itertools
from tabulate import tabulate
import logging

logger = logging.getLogger(__name__)

def print_ap_category_histogram(dataset, results):
    """
    Prints AP and AR performance for each category.
    Args:
        results: dictionary; each entry contains information for a dataset
    """
    num_classes = len(results)
    N_COLS = 14
    data = list(
        itertools.chain(
            *[
                [
                    cat,
                    out["AP2D"],
                    out["AP3D"],
                    out.get("AR2D", "-"),
                    out.get("AR3D", "-"),
                    out.get("AP3D_relative", "-"),
                    out.get("AR3D_relative", "-"),
                ]
                for cat, out in results.items()
            ]
        )
    )
    data.extend([None] * (N_COLS - (len(data) % N_COLS)))
    data = itertools.zip_longest(*[data[i::N_COLS] for i in range(N_COLS)])
    table = tabulate(
        data,
        headers=["category", "AP2D", "AP3D", "AR2D", "AR3D", "AP3D_relative", "AR3D_relative"] * (N_COLS // 7),
        tablefmt="pipe",
        numalign="left",
        stralign="center",
    )
    logger.info(
        "Performance for each of {} categories on {}:\n".format(num_classes, dataset)
        + colored(table, "cyan")
    )


def print_ap_analysis_histogram(results):
    """
    Prints AP performance for various IoU thresholds and (near, medium, far) objects.
    Args:
        results: dictionary. Each entry in results contains outputs for a dataset
    """
    N_COLS = 12
    data = []
    for name, metrics in results.items():
        data_item = [name, metrics["iters"], metrics["AP2D"], metrics["AP3D"]]
        
        # add AP3D_relative column if it exists
        if "AP3D_relative" in metrics:
            data_item.append(metrics["AP3D_relative"])
        else:
            data_item.append("-")
            
        data_item.extend([metrics["AP3D@15"], metrics["AP3D@25"], metrics["AP3D@50"], metrics["AP3D-N"], metrics["AP3D-M"], metrics["AP3D-F"],
                        metrics["AR2D"], metrics["AR3D"]])
                        
        # add AR3D_relative column if it exists
        if "AR3D_relative" in metrics:
            data_item.append(metrics["AR3D_relative"])
        else:
            data_item.append("-")
            
        data.append(data_item)
    table = tabulate(
        data,
        headers=["Dataset", "#iters", "AP2D", "AP3D", "AP3D_relative", "AP3D@15", "AP3D@25", "AP3D@50", "AP3D-N", "AP3D-M", "AP3D-F", "AR2D", "AR3D", "AR3D_relative"],
        tablefmt="grid",
        numalign="left",
        stralign="center",
    )
    logger.info(
        "Per-dataset performance analysis on test set:\n"
        + colored(table, "cyan")
    )


def print_ap_dataset_histogram(results):
    """
    Prints AP performance for each dataset.
    Args:
        results: list of dicts. Each entry in results contains outputs for a dataset
    """
    metric_names = ["AP2D", "AP3D"]
    N_COLS = 4
    data = []
    for name, metrics in results.items():
        data_item = [name, metrics["iters"], metrics["AP2D"], metrics["AP3D"]]
        data.append(data_item)
    table = tabulate(
        data,
        headers=["Dataset", "#iters", "AP2D", "AP3D"],
        tablefmt="grid",
        numalign="left",
        stralign="center",
    )
    logger.info(
        "Per-dataset performance on test set:\n"
        + colored(table, "cyan")
    )


def print_ap_omni_histogram(results):
    """
    Prints AP and AR performance for Omni3D dataset.
    Args:
        results: list of dicts. Each entry in results contains outputs for a dataset
    """
    N_COLS = 4
    data = []
    for name, metrics in results.items():
        data_item = [name, metrics["iters"], metrics["AP2D"], metrics["AP3D"]]
        
        # add AP3D_relative column if it exists
        if "AP3D_relative" in metrics and metrics["AP3D_relative"] != None:
            data_item.append(metrics["AP3D_relative"])
        else:
            data_item.append("-")
            
        data_item.extend([metrics["AR2D"], metrics["AR3D"]])
        
        # add AR3D_relative column if it exists
        if "AR3D_relative" in metrics and metrics["AR3D_relative"] != None:
            data_item.append(metrics["AR3D_relative"])
        else:
            data_item.append("-")
            
        data.append(data_item)
    table = tabulate(
        data,
        headers=["Dataset", "#iters", "AP2D", "AP3D", "AP3D_relative", "AR2D", "AR3D", "AR3D_relative"],
        tablefmt="grid",
        numalign="left",
        stralign="center",
    )
    logger.info(
        "Performance on Omni3D:\n"
        + colored(table, "magenta")
    )

def print_ap_hard_easy_for_novel(easy_metrics_formatted, hard_metrics_formatted):
    table_data = [
        ["Easy Novel", easy_metrics_formatted['AP2D'], easy_metrics_formatted['AP3D'], 
        easy_metrics_formatted['AR2D'], easy_metrics_formatted['AR3D']],
        ["Hard Novel", hard_metrics_formatted['AP2D'], hard_metrics_formatted['AP3D'],
        hard_metrics_formatted['AR2D'], hard_metrics_formatted['AR3D']]
    ]

    table = tabulate(
        table_data,
        headers=["Subset", "AP2D", "AP3D", "AR2D", "AR3D"],
        tablefmt="grid"
    )

    logger.info("Novel Categories Evaluation Results on Easy and Hard subsets:\n" + table)
