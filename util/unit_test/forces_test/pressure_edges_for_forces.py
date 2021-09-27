
#PLOTS TO CHECK IF THE PRESSURE EDGES CHOOSEN FOR THE FORCES CALCULATIONS ARE THE CORRECT ONES. JOIN TO THE END OF THE VK TIEM ITERATION


zoom_pos=[self.xD - self.D/2 -1, self.xD + self.D/2 + 1, self.Ly/2 -self.D/2 -1, self.Ly/2 + self.D/2 +1]
edges = get_exterior_edges(FORCES_MASK)
pressure = pressure.values._native.cpu().numpy()
[ [edge_hl_x, edge_hl_y], [edge_hr_x, edge_hr_y], [edge_vb_x, edge_vb_y], [edge_vt_x, edge_vt_y] ] = edges

            pressure_l = np.zeros_like(pressure)
            pressure_l[edge_hl_x, edge_hl_y] = pressure[edge_hl_x, edge_hl_y]
            plot_field(pressure_l, 
                        plot_type=['surface'],
                        options=[ ['limits', [-1,2]],
                                ['full_zoom', True],
                                ['zoom_position', zoom_pos],
                                ['edges',edges],
                                ['indeces', True],
                                ['grid', True]                                   
                                ],
                        Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                        lx='x', ly='y', lbar='mask', ltitle='', 
                        save=True, filename='./results/pressure_l.png')
            

            pressure_r = np.zeros_like(pressure)
            pressure_r[edge_hr_x, edge_hr_y] = pressure[edge_hr_x, edge_hr_y]
            plot_field(pressure_r, 
                plot_type=['surface'],
                options=[ ['limits', [-1,2]],
                          ['full_zoom', True],
                          ['zoom_position', zoom_pos],
                          ['edges',edges],
                          ['indeces', True],
                          ['grid', True]                                   
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y', lbar='mask', ltitle='', 
                save=True, filename='./results/pressure_r.png')


            pressure_t = np.zeros_like(pressure)
            pressure_t[edge_vt_x, edge_vt_y] = pressure[edge_vt_x, edge_vt_y]
            plot_field(pressure_t, 
                plot_type=['surface'],
                options=[ ['limits', [-1,2]],
                          ['full_zoom', True],
                          ['zoom_position', zoom_pos],
                          ['edges',edges],
                          ['indeces', True],
                          ['grid', True]                                   
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y', lbar='mask', ltitle='', 
                save=True, filename='./results/pressure_t.png')

            pressure_b = np.zeros_like(pressure)
            pressure_b[edge_vb_x, edge_vb_y] = pressure[edge_vb_x, edge_vb_y]
            plot_field(pressure_b, 
                plot_type=['surface'],
                options=[ ['limits', [-1, 2]],
                          ['full_zoom', True],
                          ['zoom_position', zoom_pos],
                          ['edges',edges],
                          ['indeces', True],
                          ['grid', True]                                   
                        ],
                Lx=self.Lx, Ly=self.Ly, dx=self.dx, dy=self.dy,
                lx='x', ly='y', lbar='mask', ltitle='', 
                save=True, filename='./results/pressure_b.png')