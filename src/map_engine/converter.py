import subprocess
import os
import logging

logger = logging.getLogger(__name__)

class MapConverter:
    @staticmethod
    def osm_to_sumo(input_path, output_filename):
        """
        Converts GraphML file to SUMO network format.
        """
        output_dir = "data/sumo"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        output_path = os.path.join(output_dir, f"{output_filename}.net.xml")
        
        # SUMO netconvert command
        # --geometry.remove removes unnecessary nodes for better simulation performance
        command = [
            "netconvert",
            "--osm-files", input_path, 
            "--output-file", output_path,
            "--geometry.remove", "true",        # Удаляем лишние точки на прямых
            "--roundabouts.guess", "true",      # Угадываем кольца
            "--junctions.join", "true",         # Объединяем близкие перекрестки
            "--junctions.join-dist", "10",      # Расстояние для объединения (10м)
            "--tls.join", "true",               # Группируем светофоры (критично для RL)
            "--no-internal-links", "false",     # Оставь false, если хочешь видеть очереди внутри
            "--no-turnarounds", "true",         # Убираем развороты (ускоряет расчет путей)
            "--offset.disable-normalization", "true"
        ]
        
        try:
            subprocess.run(command, check=True)
            logger.info(f"Successfully converted to SUMO: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Conversion failed: {e}")
            return None
    