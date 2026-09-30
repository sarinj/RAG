import { IsEmpty } from 'class-validator';

export class EmptyDto {
  // Keep validation metadata for this intentionally empty request body.
  @IsEmpty()
  private readonly _empty?: undefined;
}
