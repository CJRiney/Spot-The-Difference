import logging, sys

logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", 
        datefmt="%Y-%m-%d %H:%M:%S", 
        handlers=[logging.StreamHandler(sys.stdout)], 
    )

logger = logging.getLogger(__name__)