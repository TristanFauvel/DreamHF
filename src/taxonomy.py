# %%
import os
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

REMOVE_VIRUSES = True


def _newickify(node_to_children: dict, root_node: str) -> str:
    """
    Helper function for `newickify_taxonomy`.
    Given a dictionary representing the tree structure and the root node,
    returns a string in Newick format that represents the tree.

    Args:
        node_to_children (dict): A dictionary that represents the tree structure.
        root_node (str): The root node of the tree.

    Returns:
        str: A string in Newick format that represents the tree.

    Raises:
        AssertionError: If the tree is circular or if some nodes aren't in the tree.
    """
    visited_nodes = set()

    def _newick_render_node(name: str) -> str:
        assert name not in visited_nodes, "Error: The tree may not be circular!"

        if name not in node_to_children:
            # Leafs
            return F'{name}'
        else:
            # Nodes
            visited_nodes.add(name)
            children = node_to_children[name]
            children_strings = [_newick_render_node(
                child) for child in children.keys()]
            children_strings = ",".join(children_strings)
            return F'({children_strings}){name}'

    newick_string = _newick_render_node(root_node) + ';'

    # Ensure no entries in the dictionary are left unused.
    assert visited_nodes == set(
        node_to_children.keys()), "Error: some nodes aren't in the tree"

    return newick_string


def process_taxonomy() -> str:
    """
    Loads the taxtable.csv file and converts it into a tree in Newick format.
    Returns the tree as a string.

    Returns:
        str: A string that represents the taxonomy tree in Newick format.
    """
    load_dotenv()
    root = os.environ.get("root_folder")

    tax_train = pd.read_csv(root + '/train/taxtable.csv')
    df = tax_train
    taxonomy_newick = newickify_taxonomy(df)
    return taxonomy_newick


def newickify_taxonomy(df: pd.DataFrame) -> str:
    """
    Given a pandas DataFrame, generates a tree-like dictionary where the keys
    represent the nodes of the tree and the values are a dictionary representing the
    children of the node. Then, it returns the tree as a string in Newick format.

    Args:
        df (pd.DataFrame): A pandas DataFrame containing the taxonomy information.

    Returns:
        str: A string that represents the taxonomy tree in Newick format.
    """

    df.insert(0, "root", ["root"]*df.shape[0], True)

    # Convert a dataframe into tree-like dictionary

    node_to_children = {}

    # iterate over dataframe row-wise. Assuming that every row stands for one complete branch of the tree
    for row in df.itertuples():
        # remove index at position 0 and elements that contain no child ("")
        if REMOVE_VIRUSES:
            if row[2] == 'k__Viruses':
                continue
        row_list = [element for element in row[1:] if (
            element != "" and isinstance(element, str))]
        for i in range(len(row_list)-1):
            if row_list[i] in node_to_children.keys():
                # parent entry already existing
                if row_list[i+1] in node_to_children[row_list[i]].keys():
                    # entry itself already existing --> next
                    continue
                else:
                    # entry not existing --> update dict and add the connection
                    node_to_children[row_list[i]].update({row_list[i+1]: 0})
            else:
                # add the branching point
                node_to_children[row_list[i]] = {row_list[i+1]: 0}

    taxonomy_newick = _newickify(node_to_children, root_node='root')
    return taxonomy_newick


def get_taxa_list() -> List[str]:
    """
    Reads a csv file containing taxonomy information and returns an array of 
    taxonomy strings extracted from the 'Phylum' column of the file.

    Returns:
        List[str]: an array of taxonomy strings extracted from the csv file
    """
    load_dotenv()
    root = os.environ.get("root_folder")

    tax_train = pd.read_csv(root + '/train/taxtable.csv')

    df = tax_train
    df.insert(0, "root", ["root"]*df.shape[0], True)

    # iterate over dataframe row-wise. Assuming that every row stands for one complete branch of the tree
    taxa = []
    for row in df.itertuples():
        if REMOVE_VIRUSES:
            if row[2] == 'k__Viruses':
                continue
        # remove index at position 0 and elements that contain no child ("")
        row_list = [element for element in row[1:] if (
            element != "" and isinstance(element, str))]
        row_content = ';'.join(row_list[1:])
        taxa.append(row_content)
    taxa = np.array(taxa)
    return taxa


def convert_to_taxtable(input: List[str]) -> pd.DataFrame:
    """
    Takes a list of taxonomy strings and converts it into a Pandas dataframe 
    with the following columns: Domain, Phylum, Class, Order, Family, Genus, Species.

    Args:
        input (List[str]): a list of taxonomy strings

    Returns:
        pd.DataFrame: a Pandas dataframe with columns: Domain, Phylum, Class, Order, Family, Genus, Species
    """
    df = pd.DataFrame([sub.split(";") for sub in input], columns=[
                      'Domain', 'Phylum', 'Class', 'Order', 'Family', 'Genus', 'Species'])
    return df
