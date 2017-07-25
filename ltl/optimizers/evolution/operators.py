from bitstring import BitArray, BitStream
import random
import numpy as np


def bits_one_point_crossover(ind1, ind2):
    """Executes a one-point crossover on the bit-representation of input :term:`sequence` individuals.
    The individuals are first converted to a bit representation and then modified. The resulting individuals will
    respectively have the length of the other.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :returns: A tuple of two individuals.
    """

    # Get chromosomes and their spec.
    chrom1, spec1 = list_to_bitstring(ind1, get_spec=True)
    chrom2, spec2 = list_to_bitstring(ind2, get_spec=True)

    # Find the crossover point and perform crossover.
    size = min(len(chrom1), len(chrom2))
    cxpoint = random.randint(1, size - 1)

    chrom2_new = chrom2[:cxpoint] + chrom1[cxpoint:]
    chrom1_new = chrom1[:cxpoint] + chrom2[cxpoint:]

    # Convert the chromosome representation back to a phenotype
    ind1[:] = bitstring_to_list(chrom1_new, spec1)
    ind2[:] = bitstring_to_list(chrom2_new, spec2)

    return ind1, ind2


def bits_uniform_random_crossover(ind1, ind2, indpb):
    """Executes a uniform random crossover on the bit-representation of input :term:`sequence` individuals.
    The individuals are first converted to a bit representation and then modified. The resulting individuals will
    respectively have the length of the other.

    :param ind1: The first individual participating in the crossover.
    :param ind2: The second individual participating in the crossover.
    :param indpb: Bit swap probability
    :returns: A tuple of two individuals.

    Note: Assumes equal-length chromosomes
    """

    # Get chromosomes and their spec.
    chrom1, spec1 = list_to_bitstring(ind1, get_spec=True)
    chrom2, spec2 = list_to_bitstring(ind2, get_spec=True)

    chrom1_new = list(chrom1)
    chrom2_new = list(chrom2)

    # Generate swap flags
    swap = np.random.choice([0, 1], size=len(chrom1), p=[1 - indpb, indpb])
    for i, s in enumerate(swap):
        if s == 1:
            chrom1_new[i] = chrom2[i]
            chrom2_new[i] = chrom1[i]

    # Convert the chromosome representation back to a phenotype
    ind1[:] = bitstring_to_list(''.join(chrom1_new), spec1)
    ind2[:] = bitstring_to_list(''.join(chrom2_new), spec2)

    return ind1, ind2


def list_to_bitstring(individual, get_spec=False):
    """Converts an individual in the form of a list of attributes into a bit string representation, for genetic
    operator manipulations.
    Note: only supports 32-bit values (both integral and floating point types), and singly-nested lists thereof.

    :param individual: List of attributes that make up an individual.
    :param get_spec: If true, returns a spec list made of int, float or lists of them, for decoding downstream."""

    def get_bitstring(attr):
        attr_type = type(attr)

        if attr_type == np.float32:
            return BitArray(float=attr, length=32).bin
        elif attr_type == np.int32:
            return BitArray(int=attr, length=32).bin
        elif attr_type == np.float64:
            return BitArray(float=attr, length=64).bin
        elif attr_type == np.int64:
            return BitArray(int=attr, length=64).bin

    bitstring = ''
    list_spec = []
    for attr in individual:
        attr_type = type(attr)
        if attr_type in [np.int32, np.int64, np.float32, np.float64]:
            bitstring += get_bitstring(attr)
            list_spec.append(attr_type)
        elif attr_type == list:
            bitstring += ''.join(map(get_bitstring, attr))
            list_spec.append(list(map(type, attr)))
        else:
            raise NotImplementedError

    return bitstring if not get_spec else (bitstring, list_spec)


def bitstring_to_list(bitstring, list_spec):
    """Converts a string of bits into a list of attributes, according to the provided specification.

    :param bitstring: Bit-string representing the individual.
    :param list_spec: Specification of the individual-list as a list of types, which can be int, float or list of either.
    """

    def get_attr(attr_type):
        if attr_type == np.int32:
            return np.int32(bit_stream.read('int:32'))
        elif attr_type == np.float32:
            return np.float32(bit_stream.read('float:32'))
        elif attr_type == np.int64:
            return np.int64(bit_stream.read('int:64'))
        elif attr_type == np.float64:
            return np.float64(bit_stream.read('float:64'))

    bit_stream = BitStream(bin=bitstring)
    individual = []

    for attr_type in list_spec:
        if attr_type in [np.int32, np.int64, np.float32, np.float64]:
            individual.append(get_attr(attr_type))
        elif type(attr_type) == list:
            # In this case, it's a list of attribute types
            individual.append(list(map(get_attr, attr_type)))
        else:
            raise NotImplementedError

    if np.isnan(individual).any():
        print("NaN encountered. Replacing by a small float in [-1, 1].")
        individual = [random.uniform(-1, 1) if np.isnan(x) else x for x in individual]

    return individual
