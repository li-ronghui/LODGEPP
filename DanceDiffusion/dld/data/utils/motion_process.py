import torch, sys
sys.path.append('/data2/lrh/project/dance/LodgePlus/Lodge_plus_exp/DanceDiffusion')
from dld.data.utils.common.quaternion import quaternion_to_cont6d, qrot, qinv
import json

def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    # r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = rot_vel.clone()
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_root_from7(data):
    rot_vel = data[..., 4]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    # r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = rot_vel                 # rot_vel 0 为第一帧的全局角度
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 5:7]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)       # 添加起始的y轴角度
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 1]

    return r_rot_ang, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


# def recover_from_ric(data):
#     r_rot_quat, r_pos = recover_root_rot_pos(data)
#     positions = data[..., 4:-180] # 仅身体坐标
#     positions = positions.view(positions.shape[:-1] + (-1, 3))

#     '''Add Y-axis rotation to local joints'''
#     positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

#     '''Add root XZ to joints'''
#     positions[..., 0] += r_pos[..., 0:1]
#     positions[..., 2] += r_pos[..., 2:3]

#     '''Concate root and joints'''
#     positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

#     return positions
def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions

# recover global joints from root positions
# def recover_from_ric266(data, joints_num):
#     r_pos = data[..., :3]
#     r_rot_ang = data[..., 3]
#     r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
#     r_rot_quat[..., 0] = torch.cos(r_rot_ang)
#     r_rot_quat[..., 2] = torch.sin(r_rot_ang)

#     positions = data[..., 7:(joints_num - 1) * 3 + 7]
#     # new_positions = positions.clone().view(positions.shape[:-1] + (-1, 3))
#     positions = positions.view(positions.shape[:-1] + (-1, 3))

#     '''Add Y-axis rotation to local joints'''
#     positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

#     '''Add root XZ to joints'''
#     positions[..., 0] += r_pos[..., 0:1]
#     positions[..., 2] += r_pos[..., 2:3]

#     '''Concate root and joints'''
#     positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

#     return positions


def recover_from_ric266(data, joints_num):
    r_pos = data[..., :3]
    r_rot_ang = data[..., 3]
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    positions = data[..., 7:(joints_num - 1) * 3 + 7]
    # new_positions = positions.clone().view(positions.shape[:-1] + (-1, 3))
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    new_positions = positions.clone()
    new_positions[..., 0] =  positions[..., 0] + r_pos[..., 0:1]
    new_positions[..., 2] =  positions[..., 2] + r_pos[..., 2:3]

    '''Concate root and joints'''
    positions_res = torch.cat([r_pos.unsqueeze(-2), new_positions], dim=-2)

    return positions_res


def recover_from_smplx_globalinit_x_grad(data, joints_num, jsonpath=None):
    if jsonpath is None:
        jsonpath = '/data2/lrh/dataset/fine_dance/RepairedDouble/datanew/smplx_vertices655.json'
    r_pos = data[..., :3]
    r_rot_ang = data[..., 3]
    # print("torch.cos(r_rot_ang)", torch.cos(r_rot_ang).shape)
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    # print("r_rot_quat", r_rot_quat.shape)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)
    

    positions = data[..., 7:(joints_num - 1) * 3 + 7]
    new_positions = positions.clone().view(positions.shape[:-1] + (-1, 3))
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    # print("positions", positions.shape)
   
    '''Add left and right hand to wrists'''
    new_positions[..., 24:39, :] = positions[...,24:39, :] + positions[...,19:20, :]
    new_positions[..., 39:54, :] = positions[...,39:54, :] + positions[...,20:21, :]
    positions = new_positions
    if positions.shape[1] > 55:
        with open(jsonpath, 'r') as file:
            smplx_vertices655 = json.load(file)
            new_positions[..., [idx_ + 54 for idx_ in smplx_vertices655['leftHand']],:] = positions[..., [idx_ + 54 for idx_ in smplx_vertices655['leftHand']],:] + positions[..., 19:20, :]
            new_positions[..., [idx_ + 54 for idx_ in smplx_vertices655['leftHandIndex1']],:] = positions[..., [idx_ + 54 for idx_ in smplx_vertices655['leftHandIndex1']],:] + positions[..., 19:20, :]
            new_positions[..., [idx_ + 54 for idx_ in smplx_vertices655['rightHand']],:] = positions[..., [idx_ + 54 for idx_ in smplx_vertices655['rightHand']],:] + positions[..., 20:21, :]
            new_positions[..., [idx_ + 54 for idx_ in smplx_vertices655['rightHandIndex1']],:] = positions[..., [idx_ + 54 for idx_ in smplx_vertices655['rightHandIndex1']],:] + positions[..., 20:21,:]


    '''Add Y-axis rotation to local joints'''
    new_positions = qrot(qinv(r_rot_quat[..., None, :]).expand(new_positions.shape[:-1] + (4,)), new_positions)

    '''Add root XZ to local joints'''
    # positions[..., 0] += r_pos[..., 0:1]
    # positions[..., 2] += r_pos[..., 2:3]
    new_positions_ro = new_positions.clone()
    new_positions_ro[..., 0] = new_positions[..., 0] + r_pos[..., 0:1]
    new_positions_ro[..., 2] = new_positions[..., 2] + r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), new_positions_ro], dim=-2)

    return positions

# recover global joints from root velocities
# def recover_from_ric266v_grad(data, joints_num):
#     rot_vel = data[..., 4]
#     r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
#     '''Get Y-axis rotation from rotation velocity'''
#     r_rot_ang[..., 1:] = rot_vel[..., :-1]
#     r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

#     r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
#     r_rot_quat[..., 0] = torch.cos(r_rot_ang)
#     r_rot_quat[..., 2] = torch.sin(r_rot_ang)

#     r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
#     r_pos[..., 1:, [0, 2]] = data[..., :-1, 5:7]
#     '''Add Y-axis rotation to root position'''
#     r_pos = qrot(qinv(r_rot_quat), r_pos)

#     r_pos = torch.cumsum(r_pos, dim=-2)

#     r_pos[..., 1] = data[..., 1]


#     positions = data[..., 7:(joints_num - 1) * 3 + 7]
#     positions = positions.view(positions.shape[:-1] + (-1, 3))

#     '''Add Y-axis rotation to local joints'''
#     positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

#     '''Add root XZ to joints'''
#     new_positions = positions.clone()
#     new_positions[..., 0] = positions[..., 0] + r_pos[..., 0:1]
#     new_positions[..., 2] = positions[..., 2] + r_pos[..., 2:3]

#     '''Concate root and joints'''
#     positions_res = torch.cat([r_pos.unsqueeze(-2), new_positions], dim=-2)

#     return positions_res



def recover_from_ric266v_grad(data, joints_num):
    rot_vel = data[..., 4]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    # r_rot_ang = rot_vel.clone()
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    # r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), device=data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    # r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 5:7]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    # r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos_cumsum = torch.cumsum(r_pos, dim=-2)
    r_pos_combined = torch.clone(r_pos_cumsum)
    r_pos_combined[..., 1] = data[..., 1]

    positions = data[..., 7:(joints_num - 1) * 3 + 7]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    new_positions = positions.clone()
    new_positions[..., 0] = positions[..., 0] + r_pos_combined[..., 0:1]
    new_positions[..., 2] = positions[..., 2] + r_pos_combined[..., 2:3]

    '''Concate root and joints'''
    positions_res = torch.cat([r_pos_combined.unsqueeze(-2), new_positions], dim=-2)

    return positions_res


def recover_from_ricv_norotinit_grad(data, joints_num):
    # print('recover_from_ricv_norotinit_grad')
    # no rotation
    rot_vel = data[..., 4]
    # r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    # r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = rot_vel.clone()
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    # r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,), device=data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    # r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos = torch.zeros(data.shape[:-1] + (3,), device=data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 5:7]
    '''Add Y-axis rotation to root position'''
    # r_pos = qrot(qinv(r_rot_quat), r_pos)

    # r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos_cumsum = torch.cumsum(r_pos, dim=-2)
    r_pos_combined = torch.clone(r_pos_cumsum)
    r_pos_combined[..., 1] = data[..., 1]

    positions = data[..., 7:(joints_num - 1) * 3 + 7]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    new_positions = positions.clone()
    new_positions[..., 0] = positions[..., 0] + r_pos_combined[..., 0:1]
    new_positions[..., 2] = positions[..., 2] + r_pos_combined[..., 2:3]

    '''Concate root and joints'''
    positions_res = torch.cat([r_pos_combined.unsqueeze(-2), new_positions], dim=-2)

    return positions_res


def recover_from_smplx_grad(data, joints_num):
    r_pos = data[..., :3]
    r_rot_ang = data[..., 3]
    print("torch.cos(r_rot_ang)", torch.cos(r_rot_ang).shape)
    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    print("r_rot_quat", r_rot_quat.shape)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    positions = data[..., 7:(joints_num - 1) * 3 + 7]
    new_positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = positions.view(positions.shape[:-1] + (-1, 3))
   
    '''Add left and right hand to wrists'''
    new_positions[:,24:39] = positions[:,24:39] + positions[:,19:20]            # 25-40  --- 20
    new_positions[:,39:] = positions[:,39:] + positions[:,20:21]                # 40-45  --- 21

    # positions = new_positions
    '''Add Y-axis rotation to local joints'''
    new_positions = qrot(qinv(r_rot_quat[..., None, :]).expand(new_positions.shape[:-1] + (4,)), new_positions)

    '''Add root XZ to joints'''
    new_positions_ro = new_positions.clone()
    new_positions_ro[..., 0] = new_positions[..., 0] + r_pos[..., 0:1]
    new_positions_ro[..., 2] = new_positions[..., 2] + r_pos[..., 2:3]

    '''Concate root and joints'''
    positions_res = torch.cat([r_pos.unsqueeze(-2), new_positions_ro], dim=-2)

    return positions_res

'''在recover_from_smplx_v的基础上，第0帧加上绝对位置'''
def recover_from_smplx_globalinit_v(data, joints_num):          
    rot_vel = data[..., 4]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    # r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = rot_vel                 # rot_vel 0 为第一帧的全局角度
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 5:7]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)       # 添加起始的y轴角度
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 1]

    # Attention! Add global initial xz position to root
    # print('r_pos', r_pos.shape)         # T, 3
    # print('data', data.shape)
    # print('r_pos', r_pos[..., 0].shape)
    # print('data', data[..., 0].shape)
    r_pos[..., 0] = r_pos[..., 0]  + data[..., 0]
    r_pos[..., 2] = r_pos[..., 2]  + data[..., 2]

    positions = data[..., 7:(joints_num - 1) * 3 + 7]
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    # print("positions", positions.shape)
   
    '''Add left and right hand to wrists'''
    positions[..., 24:39, :] = positions[...,24:39, :] + positions[...,19:20, :]
    positions[..., 39:54, :] = positions[...,39:54, :] + positions[...,20:21, :]
    if positions.shape[1] > 55:
        with open('/data2/lrh/dataset/fine_dance/RepairedDouble/beforesplit/vertices/code/smplx_vertices655.json', 'r') as file:
            smplx_vertices655 = json.load(file)
            positions[..., [idx_ + 54 for idx_ in smplx_vertices655['leftHand']],:] = positions[..., [idx_ + 54 for idx_ in smplx_vertices655['leftHand']],:] + positions[..., 19:20, :]
            positions[..., [idx_ + 54 for idx_ in smplx_vertices655['leftHandIndex1']],:] = positions[..., [idx_ + 54 for idx_ in smplx_vertices655['leftHandIndex1']],:] + positions[..., 19:20, :]
            positions[..., [idx_ + 54 for idx_ in smplx_vertices655['rightHand']],:] = positions[..., [idx_ + 54 for idx_ in smplx_vertices655['rightHand']],:] + positions[..., 20:21, :]
            positions[..., [idx_ + 54 for idx_ in smplx_vertices655['rightHandIndex1']],:] = positions[..., [idx_ + 54 for idx_ in smplx_vertices655['rightHandIndex1']],:] + positions[..., 20:21,:]


    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to local joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def recover_from_smplx_v_grad(data, joints_num):
    rot_vel = data[..., 4]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 5:7]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)
    r_pos = torch.cumsum(r_pos, dim=-2)
    r_pos[..., 1] = data[..., 1]


    positions = data[..., 7:(joints_num - 1) * 3 + 7]
    new_positions = positions.view(positions.shape[:-1] + (-1, 3))
    positions = positions.view(positions.shape[:-1] + (-1, 3))
    # print("positions", positions.shape)
   
    '''Add left and right hand to wrists'''
    new_positions[:,24:39] = positions[:,24:39] + positions[:,19:20]
    new_positions[:,39:] = positions[:,39:] + positions[:,20:21]

    '''Add Y-axis rotation to local joints'''
    new_positions = qrot(qinv(r_rot_quat[..., None, :]).expand(new_positions.shape[:-1] + (4,)), new_positions)

    '''Add root XZ to joints'''
    new_positions_ro  = new_positions.clone()
    new_positions_ro[..., 0] = new_positions[..., 0] + r_pos[..., 0:1]
    new_positions_ro[..., 2] = new_positions[..., 2]  + r_pos[..., 2:3]
    # new_positions_ro[..., 0] += r_pos[..., 0:1]
    # new_positions_ro[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions_res = torch.cat([r_pos.unsqueeze(-2), new_positions_ro], dim=-2)

    return positions_res


if __name__ == '__main__':
    joints_num = 22
    data = torch.ones(2, 266, requires_grad=True)
    data2= torch.ones(2, 266, requires_grad=True)
    data3= torch.ones(2, 266, requires_grad=True)

    output = recover_from_ric266v_grad(data, joints_num)
    # output2 = recover_from_ric266v_grad(data2, joints_num)
    # output3 = recover_from_ric266v(data3, joints_num)

    # 定义一个简单的损失函数，例如对输出取和
    loss = output.sum()
    # loss2 = output2.sum()
    # loss3 = output3.sum()

    # 反向传播
    loss.backward()
    # loss2.backward()
    # loss3.backward()

    # 检查输入的梯度
    print(data.grad)
    '''
    data = torch.randn((10, 266), requires_grad=True).to('cuda')
    print('grad1:', data.grad)
    joints_num = 22
    # output = recover_from_ric266v_grad(data, joints_num)
    # output = data * 3 +1
    output = data.clone()

    print('output' , output.shape)
    loss = output.sum()
    loss.backward()

    # 检查梯度是否正常回传到输入
    print(data.grad)
    '''


    # data2 = torch.randn((10, 4978), requires_grad=True).to('cuda')
    # print('data2 0 :', data.grad)
    # data2.retain_grad()
    # print('data2 1 :', data.grad)
    # out2 = recover_from_smplx_globalinit_x_grad(data2, 710)
    # print('out2:', out2.requires_grad)
    # print('out2:', out2.grad)


    # x = torch.tensor([1., 2., 3.])
    # print('x_base: ', x.requires_grad)   # 正常状态：False