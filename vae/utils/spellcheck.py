from symspellpy.symspellpy import SymSpell, Verbosity  # import the module


def setup(initial_capacity=83000, prefix_length=7, max_edit_distance_dictionary=2):

    global maximum_edit_distance
    maximum_edit_distance = max_edit_distance_dictionary

    dict_path = '/home/fa6/data/symspellpy/frequency_dictionary_en_82_765.txt'
    sym_spell = SymSpell(initial_capacity, max_edit_distance_dictionary,
                         prefix_length)

    term_index = 0  # column of the term in the dictionary text file
    count_index = 1  # column of the term frequency in the dictionary text file

    if not sym_spell.load_dictionary(dict_path, term_index, count_index):
        print("Dictionary file not found")
        return

    # lookup suggestions for single-word input strings
    # input_term = "memebers"  # misspelling of "members"
    # max edit distance per lookup
    # (max_edit_distance_lookup <= max_edit_distance_dictionary)
    # max_edit_distance_lookup = 2
    # suggestion_verbosity = Verbosity.CLOSEST  # TOP, CLOSEST, ALL
    # suggestions = sym_spell.lookup(input_term, suggestion_verbosity,
    #                                max_edit_distance_lookup)
    # # display suggestion term, term frequency, and edit distance
    # for suggestion in suggestions:
    #     print("{}, {}, {}".format(suggestion.term, suggestion.count,
    #                               suggestion.distance))

    return sym_spell


def spellcheck(word, max_edit_distance=1):
    global sym_spell,maximum_edit_distance
    assert(max_edit_distance <= maximum_edit_distance)
    suggestions = sym_spell.lookup(word,  Verbosity.CLOSEST, max_edit_distance)

   # # display suggestion term, term frequency, and edit distance
    if len(suggestions) > 0:
        for i,suggestion in enumerate(suggestions):
            print("{}, {}, {}".format(suggestion.term, suggestion.count,
                                      suggestion.distance))
            if i >= 0:
                break



sym_spell = setup()
