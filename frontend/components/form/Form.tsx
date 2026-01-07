"use client";

import { ReactNode } from "react";
import { FormProvider, useForm, UseFormProps, SubmitHandler } from "react-hook-form";
import { z, ZodSchema } from "zod";

export function ZodForm<T extends ZodSchema<Record<string, unknown>>>({
  schema,
  defaultValues,
  onSubmit,
  children,
}: {
  schema: T;
  defaultValues?: UseFormProps<z.infer<T>>['defaultValues'];
  onSubmit: SubmitHandler<z.infer<T>>;
  children: ReactNode;
}) {
  const methods = useForm<z.infer<T>>({
    defaultValues,
    mode: 'onBlur',
  });

  const handleSubmit = methods.handleSubmit((values) => {
    const result = schema.safeParse(values);
    if (!result.success) {
      console.warn('Validation errors', result.error.flatten());
      return;
    }
    return onSubmit(values);
  });

  return (
    <FormProvider {...methods}>
      <form onSubmit={handleSubmit}>{children}</form>
    </FormProvider>
  );
}
