def salosensaari(root):
    pheno_df_train, pheno_df_test, readcounts_df_train, readcounts_df_test = load_data(root)

    #Aggregate the species into genus
    df_read_train = readcounts_df_train
    df_read_train.columns = [el.split('s__')[0] for el in df_read_train.columns]
    df_read_train = df_read_train.groupby(df_read_train.columns, axis=1).sum()
    df_read_train.shape
    ## Select genus-level taxonomic groups that were detected in >1% of the study participants at a within-sample relative abundance of >0.1%.
    
    total = df_read_train.sum(axis = 1)

    df_proportions_train = df_read_train.divide(total, axis='rows')


    total = df_read_train.sum(axis = 1)

    df_proportions_train = df_read_train.divide(total, axis='rows')
    selection = (df_proportions_train>0.001).mean(axis= 0) > 0.01
    df_read_train = df_read_train.loc[:, selection]
    df_read_train.shape
    #Median relative abundance of the selected genus
    relative_abundance = df_proportions_train.loc[:,selection].sum(axis= 1)
    relative_abundance.median()

    ## Centered log transformation
    from skbio.stats.composition import multiplicative_replacement

    X_mr = multiplicative_replacement(df_read_train)
    # CLR
    from skbio.stats.composition import clr
    X_clr = clr(X_mr)