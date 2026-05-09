from __future__ import annotations

import logging
import os
import subprocess

logger = logging.getLogger(__name__)


class MapConverter:
    @staticmethod
    def osm_to_sumo(input_path: str, output_filename: str, *, fast: bool = False) -> str | None:
        """OSM/XML → SUMO .net.xml. ``fast=true`` упрощает сеть и ускоряет netconvert."""

        output_dir = "data/sumo"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, f"{output_filename}.net.xml")

        internal = "true" if fast else "false"
        join_dist = "18" if fast else "10"
        roundabouts = "false" if fast else "true"

        command = [
            "netconvert",
            "--no-warnings",
            "true",
            "--osm-files",
            input_path,
            "--output-file",
            output_path,
            "--geometry.remove",
            "true",
            "--roundabouts.guess",
            roundabouts,
            "--junctions.join",
            "true",
            "--junctions.join-dist",
            join_dist,
            "--tls.join",
            "true",
            "--no-internal-links",
            internal,
            "--no-turnarounds",
            "true",
            "--offset.disable-normalization",
            "true",
        ]
        
        try:
            subprocess.run(command, check=True)
            logger.info("SUMO-сеть сохранена: %s", output_path)
            return output_path
        except Exception as e:
            logger.error("Ошибка netconvert/SUMO: %s", e)
            return None
    