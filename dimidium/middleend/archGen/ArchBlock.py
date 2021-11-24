#  *
#  *                       cloudFPGA
#  *     Copyright IBM Research, All Rights Reserved
#  *    =============================================
#  *     Created: Nov 2021
#  *     Authors: NGL
#  *
#  *     Description:
#  *        Class of the architectural blocks for DOSA
#  *        An ArchBlock is a set of consecutive engines or streams
#  *

from dimidium.middleend.archGen.ArchBrick import ArchBrick
from dimidium.lib.util import BrickImplTypes, DosaRv
from dimidium.backend.operatorSets.BaseOSG import placeholderOSG


class ArchBlock(object):

    def __init__(self, block_id=None, block_impl_type=BrickImplTypes.UNDECIDED, selected_osg=placeholderOSG,
                 brick_list=None):
        if brick_list is None:
            brick_list = []
        self.local_block_id = block_id
        self.block_uuid = block_id
        self.brick_list = brick_list
        self.block_impl_type = block_impl_type
        self.selected_osg = selected_osg
        self.build_tool = None
        self.ip_dir = None

    def __repr__(self):
        brick_id_list = []
        for br in self.brick_list:
            brick_id_list.append(br.local_brick_id)
        ret = "ArchBlock(id: {}, {}, {}, {})".format(self.block_uuid, self.selected_osg,
                                                       self.block_impl_type, brick_id_list)
        return ret

    def set_local_block_id(self, block_id):
        self.local_block_id = block_id

    def set_block_uuid(self, block_uuid):
        self.block_uuid = block_uuid

    def add_brick(self, new_brick: ArchBrick):
        if self.selected_osg == placeholderOSG:
            self.selected_osg = new_brick.selected_osg
        elif self.selected_osg != new_brick.selected_osg:
            print("[DOSA:ArchBlock:ERROR] trying to add brick with wrong OSG. Ignoring.")
            return DosaRv.ERROR
        if self.block_impl_type == BrickImplTypes.UNDECIDED:
            self.block_impl_type = new_brick.selected_impl_type
        elif self.block_impl_type != new_brick.selected_impl_type:
            print("[DOSA:ArchBlock:ERROR] trying to add brick with wrong impl_type. Ignoring.")
            return DosaRv.ERROR
        self.brick_list.append(new_brick)
        return DosaRv.OK

    def build(self, build_tool):
        self.build_tool = build_tool
        assert self.selected_osg != placeholderOSG
        self.selected_osg.build_block(self, build_tool)


