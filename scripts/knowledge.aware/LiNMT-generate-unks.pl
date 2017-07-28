for my $i (0..2499)
{
  my $line = "";
  for my $j (1..20)
  {
    $k = $i * 20 + $j;
    $line .= " "."unk$k";
  }
  $line =~ s/^\s//g;
  print $line."\n";
}
