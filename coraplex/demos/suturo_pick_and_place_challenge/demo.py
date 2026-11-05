
import os
from dataclasses import dataclass, field
from enum import StrEnum


from coraplex.plans.context_extensions import RobotAccess, StatementGrounding
from coraplex.plans.plan_transformation import PlanRewriting
from coraplex.demonstrations import RobotDemonstration
from coraplex.plans.executors import PlanExecutor, KinematicallySimulatedPlanExecutor
from coraplex.robot_plans.actions.composite.transporting import TransportAction
from coraplex.robot_plans.actions.core.robot_body import ParkArmsAction, MoveTorsoAction
from coraplex.robot_plans.plan_transformations import OpenDrawerBeforeMoveAndPickUp
from cramph.composites import Sequence
from cramph.context import ContextExtension, StatechartContext
from cramph.statechart import Statechart
from krrood.entity_query_language.factories import (
    an,
    entity,
    variable,
)
from semantic_digital_twin.specifications.kinematic_structure_entities import (
    BodySpecification,
)
from semantic_digital_twin.specifications.semantic_annotations import (
    SemanticAnnotationWithRootSpecification,
)
from semantic_digital_twin.specifications.robots import RobotSpecification
from semantic_digital_twin.specifications.worlds import WorldSpecification
from segmind.detectors.agent_event_detector_nodes import GraspDetector
from segmind.detectors.coarse_event_detector_nodes import (
    PickUpDetector,
    PlacingDetector,
)
from segmind.detectors.spatial_relation_detector_nodes import ContainmentDetector
from segmind.event_segmentation import Segmind
from semantic_digital_twin.datastructures.definitions import TorsoState
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.mixins import (
    HasRootKinematicStructureEntity,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Milk,
    Floor,
)
from krrood_spatial_math.spatial_types import HomogeneousTransformationMatrix
from krrood_spatial_math.spatial_types import Pose
from semantic_digital_twin.world import World
from semantic_digital_twin.predefined_maps.kitchen_environment import (
    KitchenEnvironment,
)
from semantic_digital_twin.collision_checking.collision_rules import (
    AllowCollisionBetweenGroups,
)
from semantic_digital_twin.specifications.connections import (
    Connection6DoFSpecification,
)

# %% what the scene is built from


class SceneFile(StrEnum):
    """
    The files the demonstration builds its scene from, under the coraplex resources.
    """
    MILK = os.path.join("objects", "milk.stl")
    @property
    def path(self) -> str:
        """
        :return: Where the file is read from.
        """
        return os.path.join(
            os.path.dirname(__file__), "..", "..", "resources", self.value
        )


class ApartmentBody(StrEnum):
    """
    The apartment's own bodies the demonstration acts on.
    """

    ROOT = "root"
    """
    The apartment's root, which marks whether the scene was already populated.
    """



@dataclass
class PlaceSettingObject:
    """
    One object of the place setting: what it is, where it starts, and where it is laid.
    """

    semantic_annotation_type: type[HasRootKinematicStructureEntity]
    """
    What the object is.

    Its body is named after it, so the plan can ask for it by type.
    """

    mesh: SceneFile
    """
    The mesh the object is shaped by.
    """

    start: HomogeneousTransformationMatrix
    """
    Where the object stands before the plan begins, in :attr:`starts_on`'s frame.
    """

    starts_on: ApartmentBody | None = None
    """
    The apartment body the object starts on.

    ``None`` starts it at the world root.
    """

    def spawn(self, world: World) -> None:
        """
        Put the object where it starts and name it among the world's annotations.

        :param world: The world to spawn into.
        """
        name = self.semantic_annotation_type.__name__
        SemanticAnnotationWithRootSpecification(
            name=name,
            semantic_annotation_type=self.semantic_annotation_type,
            root_specification=BodySpecification.mesh(name, self.mesh.path),
        ).spawn(
            world,
            parent=(
                None
                if self.starts_on is None
                else world.get_body_by_name(self.starts_on)
            ),
            parent_T_self=self.start,
        )

    def annotation_in(self, world: World) -> HasRootKinematicStructureEntity:
        """
        :param world: The world the object was spawned into.
        :return: The annotation naming the object, as the plan refers to it.
        """
        return next(
            an(
                entity(
                    variable(
                        self.semantic_annotation_type,
                        domain=world.semantic_annotations,
                    )
                )
            ).evaluate()
        )



# %% the demonstration


@dataclass
class BulletWorldDemonstration(RobotDemonstration):
    """
    The PR2 transports the milk, a bowl and a spoon onto the table in the apartment.
    """

    ros_node_name: str = "bullet_world_demo_node"
    robot_start: HomogeneousTransformationMatrix = field(
        default_factory=lambda: HomogeneousTransformationMatrix.from_xyz_rpy(
            1.1, 2.5, 0.0
        )
    )


    milk: PlaceSettingObject = field(
        default_factory=lambda: PlaceSettingObject(
            Milk,
            SceneFile.MILK,
            HomogeneousTransformationMatrix.from_xyz_rpy(3.5, 3.3, 0),
            starts_on= ApartmentBody.ROOT
        )
    )


    def scene_specification(self) -> WorldSpecification:
        """
        :return: The apartment.
        """
        return WorldSpecification(world_source=KitchenEnvironment())

    def robot_specification(self) -> RobotSpecification:
        """
        :return: The Toya, placed at :attr:`robot_start` through its odom.
        """
        return RobotSpecification(
            semantic_annotation_type=self.used_robot, world_T_odom=self.robot_start
        )

    def is_scene_populated(self, world: World) -> bool:
        return world.is_kinematic_structure_entity_in_world_by_name(ApartmentBody.ROOT)

    def populate_scene(self, world: World) -> None:
        """
        Put the place setting where it starts out, and let the world reasoner annotate
        the apartment's furniture, including the drawer the spoon lies in.
        """
        with world.modify_world():
            milk = SemanticAnnotationWithRootSpecification(
                name="milk",
                semantic_annotation_type=Milk,
                root_specification=BodySpecification.mesh(
                    "milk",
                    SceneFile.MILK.path,
                    connection_specification=Connection6DoFSpecification(),
                ),
            ).spawn(world)
        milk_box = milk.root.collision.as_bounding_box_collection_in_frame(
                milk.root
        ).bounding_box()

        milk.root.parent_connection.origin = HomogeneousTransformationMatrix.from_xyz_rpy(
                self.milk.start.x, self.milk.start.y, -milk_box.min_z, reference_frame=milk.root.parent_connection.origin.reference_frame
            )

    def build_context_extensions(self, world: World) -> list[ContextExtension]:
        return [
            RobotAccess(world.get_semantic_annotations_by_type(self.used_robot)[0]),
            StatementGrounding(),
            PlanRewriting(transformations=[OpenDrawerBeforeMoveAndPickUp()]),
        ]

    # def segment_events(self, world: World) -> AbstractContextManager:
    #     """
    #     Detect the place setting being grasped, picked up, placed and contained.
    #     """
    #     return Segmind.create_for_semantic_annotation_types(
    #         world,
    #         tuple(
    #             placed_object.semantic_annotation_type
    #             for placed_object in self.place_setting
    #         ),
    #         detector_types=(
    #             PickUpDetector,
    #             PlacingDetector,
    #             ContainmentDetector,
    #             GraspDetector,
    #         ),
    #     )

    def build_statechart(self, context: StatechartContext) -> Statechart:
        """
        Carry each object to its place on the table.
        """
        world = context.world
        robot = context.require_extension(RobotAccess).robot
        with world.modify_world():
            world.collision_manager.add_ignore_collision_rule(
                AllowCollisionBetweenGroups(
                    list(robot.bodies_with_collision),
                    [
                        floor.root
                        for floor in world.get_semantic_annotations_by_type(Floor)
                    ]))
        # left_arm = robot.left_arm
        statechart = Statechart(context=context)
        statechart.add_node(
            Sequence(
                nodes=[
                    ParkArmsAction(arms=robot.all_arms),
                    MoveTorsoAction(torso_state=TorsoState.HIGH),
                    # TransportAction.from_graspable_by_closest_grasps(
                    #     self.milk.annotation_in(world),
                    #     self.milk.target_location(world),
                    #     left_arm,
                    #     context,
                    # ),
                ]
            )
        )
        return statechart


def main(
    executor_type: type[PlanExecutor] = KinematicallySimulatedPlanExecutor,
    collision_avoidance: bool = True,
    event_segmentation: bool = True,
    debug: bool = False,
) -> None:
    """
    Run the demonstration.

    :param executor_type: The executor running the plan, which decides whether it drives
        the real robot or simulates it.
    :param collision_avoidance: Whether the statechart avoids collisions.
    :param event_segmentation: Whether SegMind segments the run into events.
    :param debug: Whether to run in debug mode, publishing every copy of the world a
        candidate is tried in.
    """

    BulletWorldDemonstration(
        used_robot=HSRB,
        executor_type=executor_type,
        collision_avoidance=collision_avoidance,
        event_segmentation=event_segmentation,
        debug=debug,
    ).run()


if __name__ == "__main__":
    main()
