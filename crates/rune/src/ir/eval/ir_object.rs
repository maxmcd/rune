use crate::collections::HashMap;
use crate::ir::eval::prelude::*;

impl Eval<&ir::IrObject> for IrInterpreter<'_> {
    type Output = IrValue;

    fn eval(&mut self, ir_object: &ir::IrObject, used: Used) -> Result<Self::Output, EvalOutcome> {
        let mut object = HashMap::with_capacity(ir_object.assignments.len());

        for (key, value) in ir_object.assignments.iter() {
            object.insert(key.as_ref().to_owned(), self.eval(value, used)?);
        }

        Ok(IrValue::Object(Shared::new(object)))
    }
}