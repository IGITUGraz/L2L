import logging

logger = logging.getLogger('optimizees.perworkerstorage')


class PerWorkerStorage:
    def __init__(self):
        logger.info("Initializing perworker storage")
        self.previous_individual = None
        self.prev_generation_number = 0

        self.current_individual = None
        self.current_generation_number = -1

    def get_current_individual(self, generation):
        logger.info("Trying to get current individual for %d", generation)
        if generation > self.current_generation_number:
            logger.info("Don't have it since current generation is %d", self.current_generation_number)
            self.previous_individual = self.current_individual
            self.prev_generation_number = self.current_generation_number
            self.current_individual = None
            assert generation == self.current_generation_number + 1
            self.current_generation_number = generation
        logger.info("Got current individual for %d", generation)
        logger.info("Individual is %s", self.current_individual)
        return self.current_individual

    def get_previous_individual(self):
        logger.info("Getting previous individual")

        assert self.current_individual is None
        return self.previous_individual

    def store_current_individual(self, generation, current_individual):
        logger.info("Storing current individual for generation %d", generation)
        assert self.current_generation_number == generation
        if not generation == -1:
            assert self.current_individual is None

        self.current_individual = current_individual
