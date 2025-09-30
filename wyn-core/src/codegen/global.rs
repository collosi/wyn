use crate::error::Result;
use log::debug;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{self, Decoration, ExecutionModel, StorageClass};
use std::collections::HashMap;

pub struct GlobalBuilder {
    // Track created builtin variables by (builtin, storage_class) -> variable_id
    builtin_variables: HashMap<(spirv::BuiltIn, StorageClass), spirv::Word>,
    // Track created location-based variables by (location, storage_class) -> variable_id
    location_variables: HashMap<(u32, StorageClass), spirv::Word>,
}

impl Default for GlobalBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalBuilder {
    pub fn new() -> Self {
        Self {
            builtin_variables: HashMap::new(),
            location_variables: HashMap::new(),
        }
    }

    /// Create or lookup a builtin variable for the given execution model
    /// Returns the variable ID if the builtin is valid for this execution model
    pub fn create_or_lookup_builtin(
        &mut self,
        builder: &mut Builder,
        builtin: spirv::BuiltIn,
        storage_class: StorageClass,
        execution_model: ExecutionModel,
    ) -> Result<Option<spirv::Word>> {
        // Check if this builtin is valid for the given execution model and storage class
        if !self.is_builtin_valid_for_execution_model(builtin, storage_class, execution_model) {
            return Ok(None);
        }

        // Check if we already created this builtin
        let key = (builtin, storage_class);
        if let Some(&var_id) = self.builtin_variables.get(&key) {
            return Ok(Some(var_id));
        }

        // Create the builtin variable
        let type_id = self.get_builtin_type(builder, builtin);
        let ptr_type = builder.type_pointer(None, storage_class, type_id);
        let var_id = builder.variable(ptr_type, None, storage_class, None);
        builder.decorate(var_id, Decoration::BuiltIn, vec![Operand::BuiltIn(builtin)]);

        // Store for future lookups
        self.builtin_variables.insert(key, var_id);

        debug!(
            "Created {:?} builtin {:?} for {:?}",
            storage_class, builtin, execution_model
        );
        Ok(Some(var_id))
    }

    /// Create or lookup a location-based variable
    pub fn create_or_lookup_location(
        &mut self,
        builder: &mut Builder,
        location: u32,
        storage_class: StorageClass,
        type_id: spirv::Word,
    ) -> Result<spirv::Word> {
        // Check if we already created this location variable
        let key = (location, storage_class);
        if let Some(&var_id) = self.location_variables.get(&key) {
            return Ok(var_id);
        }

        // Create the location variable
        let ptr_type = builder.type_pointer(None, storage_class, type_id);
        let var_id = builder.variable(ptr_type, None, storage_class, None);
        builder.decorate(
            var_id,
            Decoration::Location,
            vec![Operand::LiteralBit32(location)],
        );

        // Store for future lookups
        self.location_variables.insert(key, var_id);

        debug!(
            "Created {:?} location {} variable",
            storage_class, location
        );
        Ok(var_id)
    }

    /// Check if a builtin is valid for the given execution model and storage class
    fn is_builtin_valid_for_execution_model(
        &self,
        builtin: spirv::BuiltIn,
        storage_class: StorageClass,
        execution_model: ExecutionModel,
    ) -> bool {
        match (execution_model, storage_class) {
            (ExecutionModel::Vertex, StorageClass::Input) => {
                matches!(
                    builtin,
                    spirv::BuiltIn::VertexIndex
                        | spirv::BuiltIn::InstanceIndex
                        | spirv::BuiltIn::VertexId
                        | spirv::BuiltIn::InstanceId
                        | spirv::BuiltIn::BaseVertex
                        | spirv::BuiltIn::BaseInstance
                        | spirv::BuiltIn::DrawIndex
                )
            }
            (ExecutionModel::Vertex, StorageClass::Output) => {
                matches!(
                    builtin,
                    spirv::BuiltIn::Position
                        | spirv::BuiltIn::PointSize
                        | spirv::BuiltIn::ClipDistance
                        | spirv::BuiltIn::CullDistance
                )
            }
            (ExecutionModel::Fragment, StorageClass::Input) => {
                matches!(
                    builtin,
                    spirv::BuiltIn::FragCoord
                        | spirv::BuiltIn::FrontFacing
                        | spirv::BuiltIn::PointCoord
                        | spirv::BuiltIn::SampleId
                        | spirv::BuiltIn::SamplePosition
                        | spirv::BuiltIn::SampleMask
                        | spirv::BuiltIn::PrimitiveId
                        | spirv::BuiltIn::Layer
                        | spirv::BuiltIn::ViewportIndex
                )
            }
            (ExecutionModel::Fragment, StorageClass::Output) => {
                matches!(builtin, spirv::BuiltIn::FragDepth | spirv::BuiltIn::SampleMask)
            }
            _ => false, // Other execution models not supported yet
        }
    }

    /// Get the SPIR-V type ID for a builtin
    fn get_builtin_type(&self, builder: &mut Builder, builtin: spirv::BuiltIn) -> spirv::Word {
        match builtin {
            // Integer builtins
            spirv::BuiltIn::VertexIndex
            | spirv::BuiltIn::InstanceIndex
            | spirv::BuiltIn::VertexId
            | spirv::BuiltIn::InstanceId
            | spirv::BuiltIn::BaseVertex
            | spirv::BuiltIn::BaseInstance
            | spirv::BuiltIn::DrawIndex
            | spirv::BuiltIn::SampleId
            | spirv::BuiltIn::PrimitiveId
            | spirv::BuiltIn::Layer
            | spirv::BuiltIn::ViewportIndex => builder.type_int(32, 1),

            // Float builtins
            spirv::BuiltIn::PointSize | spirv::BuiltIn::FragDepth => builder.type_float(32),

            // Vector builtins
            spirv::BuiltIn::Position | spirv::BuiltIn::FragCoord => {
                let float_type = builder.type_float(32);
                builder.type_vector(float_type, 4)
            }
            spirv::BuiltIn::PointCoord | spirv::BuiltIn::SamplePosition => {
                let float_type = builder.type_float(32);
                builder.type_vector(float_type, 2)
            }

            // Boolean builtins
            spirv::BuiltIn::FrontFacing => builder.type_bool(),

            // Array builtins
            spirv::BuiltIn::SampleMask => {
                let int_type = builder.type_int(32, 1);
                let int_type_for_size = builder.type_int(32, 1);
                let array_size = builder.constant_bit32(int_type_for_size, 1);
                builder.type_array(int_type, array_size)
            }
            spirv::BuiltIn::ClipDistance | spirv::BuiltIn::CullDistance => {
                let float_type = builder.type_float(32);
                let int_type_for_size = builder.type_int(32, 1);
                let array_size = builder.constant_bit32(int_type_for_size, 8); // Common max
                builder.type_array(float_type, array_size)
            }

            _ => {
                // Fallback to i32 for unknown builtins
                builder.type_int(32, 1)
            }
        }
    }

    /// Get all created builtin variables for interface lists
    pub fn get_builtin_interface_variables(&self, execution_model: ExecutionModel) -> Vec<spirv::Word> {
        let vars: Vec<spirv::Word> = self
            .builtin_variables
            .iter()
            .filter_map(|((builtin, storage_class), &var_id)| {
                let is_valid =
                    self.is_builtin_valid_for_execution_model(*builtin, *storage_class, execution_model);
                debug!(
                    "Builtin {:?} {:?} for {:?}: valid={}",
                    builtin, storage_class, execution_model, is_valid
                );
                if is_valid { Some(var_id) } else { None }
            })
            .collect();
        debug!(
            "get_builtin_interface_variables for {:?} returning {} variables",
            execution_model,
            vars.len()
        );
        vars
    }

    /// Get all created location variables for interface lists
    pub fn get_location_interface_variables(&self, storage_class: StorageClass) -> Vec<spirv::Word> {
        self.location_variables
            .iter()
            .filter_map(
                |((_, sc), &var_id)| {
                    if *sc == storage_class { Some(var_id) } else { None }
                },
            )
            .collect()
    }
}
