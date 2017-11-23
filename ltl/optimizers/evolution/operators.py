from bitstring import BitArray, BitStream

import logging
import numpy as np
import random

logger = logging.getLogger("optimizers.evolution")


def list_to_bitstring(individual, bounds_spec, get_spec=False):
    """Converts an individual in the form of a list of attributes into a bit string representation, for genetic
    operator manipulations. Real numbers are encoded by first discretising the space [min_real, max_real] and indexing
    into that space by an integer of the same bit-depth. The binary representation of that integer is then used for
    genetic manipulations.
    Note: Supports both integral and floating point types, and singly-nested lists thereof.

    :param individual: List of attributes that make up an individual.
    :param get_spec: If true, returns a spec list made of int, float or lists of them, for decoding downstream.
    :param min_real: Minimum real-value bound, to use in encoding real numbers.
    :param max_real: Maximum real-value bound, to use in encoding real numbers."""

    def get_bitstring(attr, min_real, max_real):
        attr_type = type(attr)

        if attr_type == np.float32:
            encoded_int = np.uint32((attr - min_real) / (max_real - min_real) * np.iinfo(np.uint32).max)
            return BitArray(uint=encoded_int, length=32).bin
        elif attr_type == np.int32:
            return BitArray(int=attr, length=32).bin
        elif attr_type == np.float64:
            encoded_int = np.uint64((attr - min_real) / (max_real - min_real) * np.iinfo(np.uint64).max)
            return BitArray(uint=encoded_int, length=64).bin
        elif attr_type == np.int64:
            return BitArray(int=attr, length=64).bin

    bitstring = ''
    list_spec = []
    for (k, b), attr in zip(bounds_spec.items(), individual):
        attr_type = type(attr)
        if attr_type in [np.int32, np.int64, np.float32, np.float64]:
            bitstring += get_bitstring(attr, b[0], b[1])
            list_spec.append(attr_type)
        elif attr_type == list:
            bitstring += ''.join(map(get_bitstring, attr))
            list_spec.append(list(map(type, attr)))
        else:
            raise NotImplementedError

    return bitstring if not get_spec else (bitstring, list_spec)


def bitstring_to_list(bitstring, bounds_spec, list_spec):
    """Converts a string of bits into a list of attributes, according to the provided specification. This performs the
    reverse operation as list_to_bitstring. Real numbers are decoded by first converting a section of the bit string
    into an integer, which is then mapped onto a real number by a shifting and rescaling transformation.

    :param bitstring: Bit-string representing the individual.
    :param list_spec: Specification of the individual-list as a list of types, which can be int, float or list of
    either.
    :param min_real: Minimum real-value bound, to use in decoding real numbers.
    :param max_real: Maximum real-value bound, to use in decoding real numbers."""

    def get_attr(attr_type, min_real, max_real):
        if attr_type == np.int32:
            return np.int32(bit_stream.read('int:32'))
        elif attr_type == np.float32:
            decoded_int = np.uint32(bit_stream.read('uint:32'))
            return np.float32((np.float32(decoded_int) / np.iinfo(np.uint32).max) * (max_real - min_real) + min_real)
        elif attr_type == np.int64:
            return np.int64(bit_stream.read('int:64'))
        elif attr_type == np.float64:
            decoded_int = np.uint64(bit_stream.read('uint:64'))
            return np.float64((np.float64(decoded_int) / np.iinfo(np.uint64).max) * (max_real - min_real) + min_real)

    bit_stream = BitStream(bin=bitstring)
    individual = []

    for (k, b), attr_type in zip(bounds_spec.items(), list_spec):
        if attr_type in [np.int32, np.int64, np.float32, np.float64]:
            individual.append(get_attr(attr_type, b[0], b[1]))
        elif type(attr_type) == list:
            # In this case, it's a list of attribute types
            individual.append(list(map(get_attr, attr_type)))
        else:
            raise NotImplementedError

    if np.isnan(individual).any():
        logging.info("NaN encountered. Replacing by a random real number.")
        individual = [random.uniform(min_real, max_real) if np.isnan(x) else x for x in individual]

    return individual


# Mating Operators
def bits_one_point_crossover(ind1, ind2, bounds_spec):
    """Executes a one-point crossover on the binary representation of input :term:`sequence` individuals.
    The individuals are first converted to a binary representation and then modified. The resulting individuals will
    respectively have the length of the other.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param bounds_spec: Range spec ordered dictionary.
    :returns: A tuple of two offspring (input individuals are modified in-place).
    """

    # Get chromosomes and their spec.
    chrom1, spec1 = list_to_bitstring(ind1, get_spec=True, bounds_spec=bounds_spec)
    chrom2, spec2 = list_to_bitstring(ind2, get_spec=True, bounds_spec=bounds_spec)

    # Find the crossover point and perform crossover.
    size = min(len(chrom1), len(chrom2))
    cxpoint = random.randint(1, size - 1)

    chrom2_new = chrom2[:cxpoint] + chrom1[cxpoint:]
    chrom1_new = chrom1[:cxpoint] + chrom2[cxpoint:]

    # Convert the chromosome representation back to a phenotype
    ind1[:] = bitstring_to_list(chrom1_new, list_spec=spec1, bounds_spec=bounds_spec)
    ind2[:] = bitstring_to_list(chrom2_new, list_spec=spec2, bounds_spec=bounds_spec)

    return ind1, ind2


def bits_two_point_crossover(ind1, ind2, bounds_spec):
    """Executes a two-point crossover on the binary representation of input :term:`sequence` individuals.
    The individuals are first converted to a binary representation and then modified.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param bounds_spec: Range spec ordered dictionary.
    :returns: A tuple of two offspring (input individuals are modified in-place).
    """

    # Get chromosomes and their spec.
    chrom1, spec1 = list_to_bitstring(ind1, get_spec=True, bounds_spec=bounds_spec)
    chrom2, spec2 = list_to_bitstring(ind2, get_spec=True, bounds_spec=bounds_spec)

    # Find the crossover points and perform crossover.
    size = min(len(ind1), len(ind2))
    cxpoint1 = random.randint(1, size)
    cxpoint2 = random.randint(1, size - 1)
    if cxpoint2 >= cxpoint1:
        cxpoint2 += 1
    else:  # Swap the two cx points
        cxpoint1, cxpoint2 = cxpoint2, cxpoint1

    chrom1_new = chrom1[:cxpoint1] + chrom2[cxpoint1:cxpoint2] + chrom1[cxpoint2:]
    chrom2_new = chrom2[:cxpoint1] + chrom1[cxpoint1:cxpoint2] + chrom2[cxpoint2:]

    # Convert the chromosome representation back to a phenotype
    ind1[:] = bitstring_to_list(chrom1_new, list_spec=spec1, bounds_spec=bounds_spec)
    ind2[:] = bitstring_to_list(chrom2_new, list_spec=spec2, bounds_spec=bounds_spec)

    return ind1, ind2


def bits_uniform_random_crossover(ind1, ind2, swap_prob, bounds_spec):
    """Executes a uniform random crossover on the bit-representation of input :term:`sequence` individuals.
    The individuals are first converted to a bit representation and then modified. The resulting individuals will
    respectively have the length of the other.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param swap_prob: Bit swap probability.
    :param min_real: Minimum real-value bound, to use in decoding real numbers.
    :param max_real: Maximum real-value bound, to use in decoding real numbers.
    :returns: A tuple of two offspring (input individuals are modified in-place).

    Note: Assumes equal-length chromosomes
    """

    # Get chromosomes and their spec.
    chrom1, spec1 = list_to_bitstring(ind1, get_spec=True, bounds_spec=bounds_spec)
    chrom2, spec2 = list_to_bitstring(ind2, get_spec=True, bounds_spec=bounds_spec)

    chrom1_new = list(chrom1)
    chrom2_new = list(chrom2)

    # Generate swap flags
    swap = np.random.choice([0, 1], size=len(chrom1), p=[1 - swap_prob, swap_prob])
    for i, s in enumerate(swap):
        if s == 1:
            chrom1_new[i] = chrom2[i]
            chrom2_new[i] = chrom1[i]

    # Convert the chromosome representation back to a phenotype
    ind1[:] = bitstring_to_list(''.join(chrom1_new), list_spec=spec1, bounds_spec=bounds_spec)
    ind2[:] = bitstring_to_list(''.join(chrom2_new), list_spec=spec2, bounds_spec=bounds_spec)

    return ind1, ind2


# Mutation Operators
def bits_random_bitflip_mutation(ind, flip_prob, bounds_spec):
    """Executes a random bitflip mutation on the chromosome of input :term:`sequence` individual.
    The individual is first converted to a bit representation and then modified.

    :param ind: The individual to be mutated.
    :param flip_prob: Bit flip probability.
    :param min_real: Minimum real-value bound, to use in decoding real numbers.
    :param max_real: Maximum real-value bound, to use in decoding real numbers.
    :returns: A tuple with one mutated individual (which is also modified in-place).
    """

    # Get chromosomes and spec.
    chrom, spec = list_to_bitstring(ind, get_spec=True, bounds_spec=bounds_spec)

    chrom_new = list(chrom)

    # Generate flip flags
    flip = np.random.choice([0, 1], size=len(chrom), p=[1 - flip_prob, flip_prob])
    for i, s in enumerate(flip):
        if s == 1:
            chrom_new[i] = '1' if chrom_new[i] == '0' else '0'

    # Convert the chromosome representation back to a phenotype
    ind[:] = bitstring_to_list(''.join(chrom_new), list_spec=spec, bounds_spec=bounds_spec)

    return ind,
