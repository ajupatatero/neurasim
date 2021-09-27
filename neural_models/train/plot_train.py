import os
import torch
import copy

import matplotlib
if 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import numpy as np
import pdb

from engines.phi import math, field
from engines.phi.field import GeometryMask, AngularVelocity, Grid, StaggeredGrid, divergence, CenteredGrid, spatial_gradient, where, HardGeometryMask
from engines.phi.geom import union

from .util_train import convert_phi_to_torch

def plot_train(print_list, batch_idx, epoch, data, target, flags, m_path,
                 pressure, velocity, div_out, div_in, DOMAIN, config, save_or_show, loss_list):

    div_out_t, out_p_t, out_U_t = convert_phi_to_torch(velocity, pressure, div_out)
    print_list = [batch_idx*len(data), epoch]
    im_path = os.path.join(m_path, 'Images')
    if not os.path.exists(im_path):
        os.makedirs(im_path)
    filename_p = 'output_p_{0:05d}_ep_{1:03d}.png'.format(*print_list)
    filename_vel = 'output_v_{0:05d}_ep_{1:03d}.png'.format(*print_list)
    filename_div = 'output_div_{0:05d}_ep_{1:03d}.png'.format(*print_list)

    file_plot_p = os.path.join(im_path, filename_p)
    file_plot_vel = os.path.join(im_path, filename_vel)
    file_plot_div = os.path.join(im_path, filename_div)


    with torch.no_grad():

        # Now solve with torch to debug
        obstacles = ()
        active = DOMAIN.grid(HardGeometryMask(~union(*[obstacle.geometry for obstacle in obstacles])), extrapolation=DOMAIN.boundaries['active_extrapolation'])
        accessible = DOMAIN.grid(active, extrapolation=DOMAIN.boundaries['accessible_extrapolation'])
        hard_bcs = field.stagger(accessible, math.minimum, DOMAIN.boundaries['accessible_extrapolation'], type=type(velocity))
        def laplace(p):
            grad = spatial_gradient(p, type(velocity))
            grad *= hard_bcs
            grad = grad.with_(extrapolation=DOMAIN.boundaries['near_vector_extrapolation'])
            div = divergence(grad)
            lap = where(active, div, p)
            return lap

        pressure_guess = None
        solve_params= math.LinearSolve(None, 1e-5)
        pressure_guess = pressure_guess if pressure_guess is not None else DOMAIN.scalar_grid(0)
        converged, pressure_CG, iterations = field.solve(laplace, y=div_in, x0=pressure_guess, solve_params=solve_params, constants=[active, hard_bcs])
        if math.all_available(converged) and not math.all(converged):
            raise AssertionError(f"pressure solve did not converge after {iterations} iterations\nResult: {pressure_CG.values}")

        p_target = pressure_CG.values._native.transpose(-1, -2).unsqueeze(1).unsqueeze(1)#.repeat(1,4,1,1,1)        

        p_mean_NN = torch.mean(out_p_t[0])
        p_mean_CG = torch.mean(p_target[0])

        p_target = torch.cat((p_target, p_target, p_target, p_target), dim =1)[0].unsqueeze(0)

        plotField(out=[out_p_t[0].unsqueeze(0).unsqueeze(0).unsqueeze(0) - p_mean_NN,
                            out_U_t[0].unsqueeze(0).unsqueeze(2),
                            div_out_t[0].unsqueeze(0).unsqueeze(0).unsqueeze(0)],
                        tar=p_target - p_mean_CG,
                        flags=flags[0].unsqueeze(0),
                        loss=loss_list,
                        mconf=config,
                        epoch=epoch,
                        filename=file_plot_p,
                        save=save_or_show,
                        plotPres=True,
                        plotVel=False,
                        plotDiv=False,
                        title=False,
                        x_slice=104)
        plotField(out=[out_p_t[0].unsqueeze(0).unsqueeze(0).unsqueeze(0),
                            out_U_t[0].unsqueeze(0).unsqueeze(2),
                            div_out_t[0].unsqueeze(0).unsqueeze(0).unsqueeze(0)],
                        tar=target[0].unsqueeze(0),
                        flags=flags[0].unsqueeze(0),
                        loss=loss_list,
                        mconf=config,
                        epoch=epoch,
                        filename=file_plot_vel,
                        save=save_or_show,
                        plotPres=False,
                        plotVel=True,
                        plotDiv=False,
                        title=False,
                        x_slice=104)
        plotField(out=[out_p_t[0].unsqueeze(0).unsqueeze(0).unsqueeze(0),
                            out_U_t[0].unsqueeze(0).unsqueeze(2),
                            div_out_t[0].unsqueeze(0).unsqueeze(0).unsqueeze(0)],
                        tar=target[0].unsqueeze(0),
                        flags=flags[0].unsqueeze(0),
                        loss=loss_list,
                        mconf=config,
                        epoch=epoch,
                        filename=file_plot_div,
                        save=save_or_show,
                        plotPres=False,
                        plotVel=False,
                        plotDiv=True,
                        title=False,
                        x_slice=104)


def plotField(out, tar, flags, loss, mconf, epoch=None, filename=None, save=False,
        plotGraphs=True, plotPres=True, plotVel=True, plotDiv=True, title=True, **kwargs):

    x_slice = None
    y_slice = None
    if kwargs:
        assert len(kwargs.keys()) == 1, 'Only one slice allowed'
        if 'x_slice' in kwargs:
            x_slice = kwargs['x_slice']
        elif 'y_slice' in kwargs:
            y_slice = kwargs['y_slice']
        else:
            assert False, 'kwargs must be either x_slice or y_slice'

    else:
        if plotGraphs:
            print('You must specify either a x-slice or y-slice')

    target = tar.clone()
    output_p = out[0].clone()
    output_U = out[1].clone()
    p_out = output_p
    p_tar = target[:,0].unsqueeze(1)
    U_norm_out = torch.zeros_like(p_out)
    U_norm_tar = torch.zeros_like(p_tar)

    Ux_out = output_U[:,0].unsqueeze(1)
    Uy_out = output_U[:,1].unsqueeze(1)
    Ux_tar = target[:,1].unsqueeze(1)
    Uy_tar = target[:,2].unsqueeze(1)
    torch.norm(output_U, p=2, dim=1, keepdim=True, out=U_norm_out)
    torch.norm(target[:,1:3], p=2, dim=1, keepdim=True, out=U_norm_tar)

    div = out[2].clone()

    err_p = (p_out - p_tar)**2
    err_Ux = (Ux_out - Ux_tar)**2
    err_Uy = (Uy_out - Uy_tar)**2
    err_U_norm = (U_norm_out - U_norm_tar)**2
    err_div = (div)**2

    max_val_p_tar = torch.max(p_tar).cpu().data.numpy()
    max_val_p_out = torch.max(p_out).cpu().data.numpy()
    min_val_p_tar = torch.min(p_tar).cpu().data.numpy()
    min_val_p_out = torch.min(p_out).cpu().data.numpy()
    diff_val_tar = abs(max_val_p_tar - min_val_p_tar)

    max_val_Ux = np.maximum(torch.max(Ux_out).cpu().data.numpy(), \
                         torch.max(Ux_tar).cpu().data.numpy() )
    min_val_Ux = np.minimum(torch.min(Ux_out).cpu().data.numpy(), \
                         torch.min(Ux_tar).cpu().data.numpy())
    max_val_Uy = np.maximum(torch.max(Uy_out).cpu().data.numpy(), \
                         torch.max(Uy_tar).cpu().data.numpy() )
    min_val_Uy = np.minimum(torch.min(Uy_out).cpu().data.numpy(), \
                         torch.min(Uy_tar).cpu().data.numpy())
    max_val_U_norm = np.maximum(torch.max(U_norm_out).cpu().data.numpy(), \
                         torch.max(U_norm_tar).cpu().data.numpy() )
    min_val_U_norm = np.minimum(torch.min(U_norm_out).cpu().data.numpy(), \
                         torch.min(U_norm_tar).cpu().data.numpy() )
    max_err_p = torch.max(err_p).cpu().data.numpy()
    max_err_Ux = torch.max(err_Ux).cpu().data.numpy()
    max_err_Uy = torch.max(err_Uy).cpu().data.numpy()
    max_err_U_norm = torch.max(err_U_norm).cpu().data.numpy()

    max_div = torch.max(div).cpu().data.numpy()
    min_div = torch.min(div).cpu().data.numpy()

    p_tar_line = p_tar.clone()
    p_out_line = p_out.clone()
    U_tar_line = U_norm_tar.clone()
    U_out_line = U_norm_out.clone()
    div_line = div.clone()
    mask = flags.eq(2)
    p_tar.masked_fill_(mask, 100)
    p_out.masked_fill_(mask, 100)
    Ux_tar.masked_fill_(mask, 0)
    Ux_out.masked_fill_(mask, 0)
    Uy_tar.masked_fill_(mask, 0)
    Uy_out.masked_fill_(mask, 0)
    U_norm_tar.masked_fill_(mask, 100)
    U_norm_out.masked_fill_(mask, 100)
    div.masked_fill_(mask, 100)

    err_p.masked_fill_(mask, 100)
    err_Ux.masked_fill_(mask, 100)
    err_Uy.masked_fill_(mask, 100)
    err_U_norm.masked_fill_(mask, 100)
    err_div.masked_fill_(mask, 100)

    p_tar_line_np =torch.squeeze(p_tar_line).cpu().data.numpy()
    p_out_line_np =torch.squeeze(p_out_line).cpu().data.numpy()
    U_tar_line_np =torch.squeeze(U_tar_line).cpu().data.numpy()
    U_out_line_np =torch.squeeze(U_out_line).cpu().data.numpy()
    div_line_np =torch.squeeze(div_line).cpu().data.numpy()

    p_tar_np =torch.squeeze(p_tar).cpu().data.numpy()
    p_out_np =torch.squeeze(p_out).cpu().data.numpy()
    Ux_tar_np =torch.squeeze(Ux_tar).cpu().data.numpy()
    Ux_out_np =torch.squeeze(Ux_out).cpu().data.numpy()
    Uy_tar_np =torch.squeeze(Uy_tar).cpu().data.numpy()
    Uy_out_np =torch.squeeze(Uy_out).cpu().data.numpy()
    U_norm_tar_np =torch.squeeze(U_norm_tar).cpu().data.numpy()
    U_norm_out_np =torch.squeeze(U_norm_tar).cpu().data.numpy()
    div_np =torch.squeeze(div).cpu().data.numpy()
    err_p_np =torch.squeeze(err_p).cpu().data.numpy()
    err_Ux_np =torch.squeeze(err_Ux).cpu().data.numpy()
    err_Uy_np =torch.squeeze(err_Uy).cpu().data.numpy()
    err_U_norm_np = torch.squeeze(err_U_norm).cpu().data.numpy()
    err_div_np =torch.squeeze(err_div).cpu().data.numpy()

    title_list = []
    numLoss = 0
    if mconf['pL2Lambda'] > 0:
        numLoss +=1

    if mconf['divL2Lambda'] > 0:
        numLoss +=1

    if mconf['pL1Lambda'] > 0:
        numLoss +=1

    if mconf['divL1Lambda'] > 0:
        numLoss +=1

    if ('divLongTermLambda' in mconf) and mconf['divLongTermLambda'] > 0:
        numLoss +=1

    if mconf['pL2Lambda'] > 0:
        title_list.append(str(mconf['pL2Lambda']) + ' * L2(p)')

    if mconf['divL2Lambda'] > 0:
        title_list.append(str(mconf['divL2Lambda']) + ' * L2(div)')

    if mconf['pL1Lambda'] > 0:
        title_list.append(str(mconf['pL1Lambda']) + ' * L1(p)')

    if mconf['divL1Lambda'] > 0:
        title_list.append(str(mconf['divL1Lambda']) + ' * L1(div)')

    if ('divLongTermLambda' in mconf) and (mconf['divLongTermLambda'] > 0):
        title_list.append(str(mconf['divLongTermLambda']) + ' * L2(LongTermDiv)')

    title = ''
    for string in range(0, numLoss - 1):
        title += title_list[string] + ' + '
    title += title_list[numLoss-1]

    my_cmap = cm.jet
    cmap = copy.copy(cm.get_cmap("jet"))
    cmap.set_over('white')
    cmap_1 = copy.copy(cm.get_cmap("jet"))
    cmap_1.set_under('white')

    nrow = 0
    height_ratios = []
    if plotGraphs:
        width_ratios = [1,1,1,1]
        ncol = 4
    else:
        width_ratios = [1,1,1]
        ncol = 3
    if plotPres:
        nrow += 1
        height_ratios.append(1)
    if plotVel:
        nrow += 1
        height_ratios.append(1)
    if plotDiv:
        nrow += 1
        height_ratios.append(1)

    matplotlib.rc('text')
    px, py = 1580, 950
    dpi = 100
    figx = px / dpi
    figy = py / dpi
    fig = plt.figure(figsize=(figx, figy), dpi=dpi)
    if not save:
        plt.switch_backend('QT5Agg')
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
    gs = gridspec.GridSpec(nrow, ncol,
                     width_ratios=width_ratios,
                     height_ratios=height_ratios,
                     wspace=0.2, hspace=0.2,
                     top=0.91, bottom=0.05,
                     left=0.05, right=0.95)

    if title:
        if epoch is not None:
            fig.suptitle(r'$\bf{FluidNet\ output}$' + ' - Loss = ' + title + ' - ' + r'$\bf{' + 'Epoch : ' + str(epoch) + '}$')
        else:
            fig.suptitle(r'$\bf{FluidNet\ output}$' ' - Loss = ' + title)

    it_row = 0
    s = np.linspace(0, 127)
    if x_slice is not None:
        f_s = x_slice
    if y_slice is not None:
        f_s = y_slice


    if plotPres:
        it_col = 0
        if plotGraphs:
            if x_slice is not None:
                P_out = p_out_line_np[f_s,:]
                P_tar = p_tar_line_np[f_s,:]
            if y_slice is not None:
                P_out = p_out_line_np[:,f_s]
                P_tar = p_tar_line_np[:,f_s]

            ax = plt.subplot(gs[it_row,it_col])
            ax.set_title('P')
            ax.plot(P_out, label = 'Output')
            ax.plot(P_tar, label = 'Target')
            x0,x1 = ax.get_xlim()
            y0,y1 = ax.get_ylim()
            ax.set_aspect((x1-x0)/(y1-y0))
            ax.legend()
            ax.set_ylabel('Pressure')
            it_col += 1

        ax = plt.subplot(gs[it_row,it_col])
        ax.set_title('Presure (target)')
        ax.axis('off')
        ax.imshow(p_tar_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_val_p_tar,max_val_p_tar])
        if x_slice is not None:
            ax.plot(s, [f_s for i in range(len(s))])
        if y_slice is not None:
            ax.plot([f_s for i in range(len(s))], s)
        it_col += 1

        ax = plt.subplot(gs[it_row, it_col])
        ax.set_title('Pressure (predicted)')
        ax.axis('off')
        ax.imshow(p_out_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_val_p_out,max_val_p_out])
        it_col += 1

        ax = plt.subplot(gs[it_row, it_col])
        ax.set_title('error P')
        ax.axis('off')
        ax.imshow(err_p_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[0,max_err_p])
        it_row += 1


    if plotVel:
        skip = 4
        scale = 0.1
        scale_units = 'xy'
        angles = 'xy'
        headwidth = 3
        headlength = 3
        Ux_tar_np_adm = Ux_tar_np / np.max(np.sqrt(Ux_tar_np **2 + Uy_tar_np **2))
        Uy_tar_np_adm = Uy_tar_np / np.max(np.sqrt(Ux_tar_np **2 + Uy_tar_np **2))
        Ux_out_np_adm = Ux_out_np / np.max(np.sqrt(Ux_out_np **2 + Uy_out_np **2))
        Uy_out_np_adm = Uy_out_np / np.max(np.sqrt(Ux_out_np **2 + Uy_out_np **2))
        it_col = 0

        if plotGraphs:
            if x_slice is not None:
                U_out = U_out_line_np[f_s,:]
                U_tar = U_tar_line_np[f_s,:]
            if y_slice is not None:
                U_out = U_out_line_np[:,f_s]
                U_tar = U_tar_line_np[:,f_s]
            ax = plt.subplot(gs[it_row,it_col])
            ax.set_title('Velocity')
            ax.plot(U_out, label = 'Output')
            ax.plot(U_tar, label = 'Target')
            x0,x1 = ax.get_xlim()
            y0,y1 = ax.get_ylim()
            ax.set_aspect((x1-x0)/(y1-y0))
            ax.legend()
            ax.set_ylabel('V')
            it_col += 1

        ax = plt.subplot(gs[it_row, it_col])
        ax.set_title('Vel-norm (target)')
        ax.axis('off')
        X, Y = np.linspace(0, 127, num=128), np.linspace(0, 127, num=128)
        ax.imshow(U_norm_tar_np, cmap=my_cmap, origin='lower',
                interpolation='none', clim=[min_val_U_norm,max_val_U_norm])
        ax.quiver(X[::skip], Y[::skip],
                Ux_tar_np_adm[::skip, ::skip], Uy_tar_np_adm[::skip, ::skip],
                scale_units=scale_units,
                angles=angles,
                headwidth=headwidth, headlength=headlength, scale=scale,
                color='pink')
        if x_slice is not None:
            ax.plot(s, [f_s for i in range(len(s))])
        if y_slice is not None:
            ax.plot([f_s for i in range(len(s))], s)
        it_col += 1

        ax = plt.subplot(gs[it_row, it_col])
        ax.set_title('Vel-norm (predicted)')
        ax.axis('off')
        ax.imshow(U_norm_out_np, cmap=my_cmap, origin='lower',
                interpolation='none', clim=[min_val_U_norm,max_val_U_norm])
        #ax.quiver(X[::skip], Y[::skip],
        #        Ux_out_np_adm[::skip, ::skip], Uy_out_np_adm[::skip, ::skip],
        #        scale_units=scale_units,
        #        angles=angles,
        #        headwidth=headwidth, headlength=headlength, scale=scale,
        #        color='pink')
        it_col += 1

        ax = plt.subplot(gs[it_row, it_col])
        ax.set_title('error U')
        ax.axis('off')
        ax.imshow(err_U_norm_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[0,max_err_U_norm])
        it_row += 1

    if plotDiv:
        it_col = 0
        if plotGraphs:
            if x_slice is not None:
                div_out = div_line_np[f_s,:]
            if y_slice is not None:
                div_out = div_line_np[:,f_s]
            ax = plt.subplot(gs[it_row,it_col])
            ax.set_title('Divergence')
            ax.plot(div_out, label = 'Output')
            ax.plot(s, [0 for i in range(len(s))], label = 'Target')
            x0,x1 = ax.get_xlim()
            y0,y1 = ax.get_ylim()
            ax.set_aspect((x1-x0)/(y1-y0))
            ax.legend()
            ax.set_ylabel('div')
            it_col += 1

        ax = plt.subplot(gs[it_row, it_col])
        ax.set_title('Divergence')
        ax.axis('off')
        ax.imshow(div_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[min_div,max_div])
        if x_slice is not None:
            ax.plot(s, [f_s for i in range(len(s))])
        if y_slice is not None:
            ax.plot([f_s for i in range(len(s))], s)
        it_col += 1

        ax = plt.subplot(gs[it_row, it_col])
        ax.set_title('Divergence error')
        ax.axis('off')
        ax.imshow(err_div_np,cmap=my_cmap, origin='lower', interpolation='none', \
                clim=[0,max_div**2])
        it_col += 1

        ax = plt.subplot(gs[it_row, it_col], aspect='equal')
        ax.set_title('Losses')
        ax.axis('off')
        for y, loss_str, val in zip(np.linspace(0.2,0.95,6),
                            ['L2(p):', 'L2(div):', 'L2(LTDiv):',
                            'L1(p):', 'L1(div):'],
                            loss[1:]):
            ax.text(0.1, y, ('{0:10} {1:.5f}').format(loss_str,val),
                    fontsize=12,
                    family='monospace')
        loss_str = 'Total:'
        ax.text(0.1, 0.05, ('{0:10} {1:.5f}').format(loss_str, loss[0]),
                fontsize=12,
                weight='bold',
                family='monospace')
        it_row += 1

    #fig.colorbar(imP, cax=cbar_ax_p, orientation='vertical')
    #cbar_ax_U = fig.add_axes([0.375, 0.45, 0.01, 0.33])
    #fig.colorbar(imU, cax=cbar_ax_U, orientation='vertical')
    #fig.set_size_inches((11, 11), forward=False)
    if save:
        #print('Saving output')
        fig.savefig(filename)
    else:
        #print('Printing output')
        plt.show(block=True)
