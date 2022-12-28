# %%
import math
import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv

REMOVE_VIRUSES = True

def _newickify(node_to_children, root_node) -> str:
    visited_nodes = set()

    def _newick_render_node(name) -> str:
        #assert name not in visited_nodes, "Error: The tree may not be circular!"

        if name not in node_to_children:
            # Leafs
            return F'{name}'
        else:
            # Nodes
            visited_nodes.add(name)
            children = node_to_children[name]
            children_strings = [_newick_render_node(child) for child in children.keys()]
            children_strings = ",".join(children_strings)
            return F'({children_strings}){name}'

    newick_string = _newick_render_node(root_node) + ';'

    # Ensure no entries in the dictionary are left unused.
    #assert visited_nodes == set(node_to_children.keys()), "Error: some nodes aren't in the tree"

    return newick_string

def process_taxonomy():
    load_dotenv()
    root = os.environ.get("root_folder")

    tax_train = pd.read_csv(root + '/train/taxtable.csv')
    # %%
    df = tax_train
    taxonomy_newick = newickify_taxonomy(df)
    return taxonomy_newick
    
def newickify_taxonomy(df):       
    df.insert(0, "root", ["root"]*df.shape[0], True)

    # Convert a dataframe into tree-like dictionary

    node_to_children = {}

    #iterate over dataframe row-wise. Assuming that every row stands for one complete branch of the tree
    for row in df.itertuples():
        #remove index at position 0 and elements that contain no child ("")
        if REMOVE_VIRUSES:
            if row[2] == 'k__Viruses':
                continue
        row_list = [element for element in row[1:] if (element != "" and isinstance(element, str))]
        for i in range(len(row_list)-1):
            if row_list[i] in node_to_children.keys():
                #parent entry already existing
                if row_list[i+1] in node_to_children[row_list[i]].keys():
                    #entry itself already existing --> next
                    continue
                else:
                    #entry not existing --> update dict and add the connection
                    node_to_children[row_list[i]].update({row_list[i+1]:0})
            else:
                #add the branching point
                node_to_children[row_list[i]] = {row_list[i+1]:0}

    taxonomy_newick = _newickify(node_to_children, root_node='root')
    return taxonomy_newick


def get_taxa_list():
    load_dotenv()
    root = os.environ.get("root_folder")

    tax_train= pd.read_csv(root + '/train/taxtable.csv')
    # %%
    df = tax_train
    df.insert(0, "root", ["root"]*df.shape[0], True)

    #iterate over dataframe row-wise. Assuming that every row stands for one complete branch of the tree
    taxa = []
    for row in df.itertuples():
        if REMOVE_VIRUSES:
            if row[2] == 'k__Viruses':
                continue
        #remove index at position 0 and elements that contain no child ("")
        row_list = [element for element in row[1:] if (element != "" and isinstance(element, str))]
        row_content = ';'.join(row_list[1:])
        taxa.append(row_content)
    taxa = np.array(taxa)
    return taxa

def convert_to_taxtable(input):
    df = pd.DataFrame([sub.split(";") for sub in input], columns= ['Domain','Phylum','Class','Order','Family','Genus','Species'])
    return df