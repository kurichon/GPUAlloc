   for dset in n_graphs:
        set_adj = [[] for _ in n_graphs[dset]]
        set_features = [[] for _ in n_graphs[dset]]
        set_graph_labels = [[] for _ in n_graphs[dset]]

        t = tqdm(total=np.sum(n_graphs[dset]), desc=dset, leave=True, unit=' graphs')
        for batch, batch_size in enumerate(n_graphs[dset]):

            for i in range(batch_size):
                model_name = data_dict[dset][i + batch * batch_size]
                g = build_graph.build_nx_graph(pbtxt_folder_path, nodetype_file, model_name)
                g.remove_nodes_from(list(nx.isolates(g)))
                #nx.draw(g)
                #plt.show()
                if nx.number_connected_components(g) > 1:
                    g = g.subgraph(max(nx.connected_components(g), key=len)).copy()

                grouped_g = build_graph.group_nx_graph(g, n_of_groups)
                grouped_g.remove_edges_from(nx.selfloop_edges(grouped_g))
                #nx.draw_networkx(grouped_g)
                #plt.show()
                nodes = list(grouped_g)
                adj_temp = nx.to_numpy_array(grouped_g, nodes)
                t.update(1)

                f1 = np.array([grouped_g.nodes[i]['tensorsize'] for i in range(grouped_g.number_of_nodes())])
                f2 = np.array([grouped_g.nodes[i]['n_of_grouped_nodes'] for i in range(grouped_g.number_of_nodes())])
                f3 = np.array([grouped_g.nodes[i]['node_type'] for i in range(grouped_g.number_of_nodes())])
                features_temp = np.stack([f1, f2, f3], axis=1)

                graph_labels_temp = np.asarray(model_graph_label_dict[model_name]).flatten()
                set_adj[batch].append(adj_temp)
                set_features[batch].append(features_temp)
                
                
                
